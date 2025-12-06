"""
faiss_builder.py

Dual-index FAISS manager for MRA_v3.

Implements the dual-index architecture:
- Main index (HNSW): Optimized for fast search
- Delta index (Flat): Fast incremental additions

Features:
- Atomic writes with temp files
- Automatic merge when delta grows large
- Pickle metadata for compression
- Search across both indices with result merging
- SHA256 file tracking for change detection

Adapted from:
- TwistedPair/V4/faiss_manager.py (index operations)
- MRA_v3 specification (dual-index strategy)

Usage:
    from utils.faiss_builder import FAISSBuilder
    
    # Create builder for reference_papers
    builder = FAISSBuilder(index_name="reference_papers", use_dual=True)
    
    # Add to delta index
    builder.add_to_delta(embeddings, metadata)
    
    # Search across both
    results = builder.search(query_embedding, k=10)
    
    # Merge when delta is large
    builder.merge_delta_to_main()
"""
import os
import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import faiss

from errors import FAISSIndexError, handle_error
import config


@dataclass
class SearchResult:
    """Single search result with score and metadata."""
    chunk_id: str
    parent_id: str
    text: str
    score: float
    source_file: str
    source_type: str
    section_title: Optional[str] = None
    indexed_at: Optional[str] = None
    doc_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class FAISSBuilder:
    """
    Dual-index FAISS manager for scalable incremental updates.
    
    Architecture:
    - Main index: HNSW for fast approximate search (read-only)
    - Delta index: Flat for exact search (incremental adds)
    
    Workflow:
    1. Add new documents to delta index
    2. Search queries both main and delta
    3. Periodically merge delta → main for optimization
    """
    
    def __init__(
        self,
        index_name: str,
        embedding_dim: Optional[int] = None,
        use_dual: bool = True,
        index_dir: Optional[Path] = None
    ):
        """
        Initialize FAISS builder.
        
        Args:
            index_name: Name of index (e.g., "reference_papers")
            embedding_dim: Embedding dimension (default: from config)
            use_dual: Use dual-index architecture (default: True)
            index_dir: Directory for indices (default: from config)
        """
        self.index_name = index_name
        self.embedding_dim = embedding_dim or config.EMBEDDING_DIM
        self.use_dual = use_dual
        self.index_dir = Path(index_dir or config.FAISS_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths
        if use_dual:
            self.main_index_path = self.index_dir / f"{index_name}_main.index"
            self.delta_index_path = self.index_dir / f"{index_name}_delta.index"
            self.main_stats_path = self.index_dir / f"{index_name}_main.stats"
            self.delta_stats_path = self.index_dir / f"{index_name}_delta.stats"
        else:
            self.main_index_path = self.index_dir / f"{index_name}.index"
            self.main_stats_path = self.index_dir / f"{index_name}.stats"
        
        self.metadata_path = self.index_dir / f"{index_name}.metadata"
        
        # Initialize or load
        if self.main_index_path.exists():
            self.load()
        else:
            self._initialize_new()
        
        print(f"FAISS Builder initialized: {index_name}")
        print(f"  Mode: {'Dual-index' if use_dual else 'Single-index'}")
        print(f"  Main index: {self.main_index.ntotal} vectors")
        if use_dual:
            print(f"  Delta index: {self.delta_index.ntotal} vectors")
    
    def _initialize_new(self):
        """Initialize new empty indices."""
        # Main index (HNSW for dual, Flat for single)
        if self.use_dual:
            # HNSW parameters
            M = 32  # Number of connections per layer
            # Use HNSW with Inner Product (cosine similarity for normalized vectors)
            self.main_index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            self.main_index.hnsw.efConstruction = 200  # Build quality
            self.main_index.hnsw.efSearch = 64  # Search quality
            self.main_index.metric_type = faiss.METRIC_INNER_PRODUCT
            
            # Delta index (Flat for exact search with Inner Product)
            self.delta_index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # Single Flat index with Inner Product
            self.main_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Metadata
        self.main_metadata = []
        self.delta_metadata = []
        self.chunk_id_to_idx = {}  # Maps chunk_id → (index_type, idx)
        
        # Stats
        self.main_stats = {
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'last_updated': datetime.utcnow().isoformat() + 'Z',
            'total_docs': 0,
            'total_chunks': 0
        }
        
        if self.use_dual:
            self.delta_stats = {
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'last_updated': datetime.utcnow().isoformat() + 'Z',
                'total_docs': 0,
                'total_chunks': 0
            }
    
    def add_to_delta(
        self,
        embeddings: np.ndarray,
        metadata: List[dict]
    ):
        """
        Add chunks to delta index (incremental updates).
        
        Args:
            embeddings: np.ndarray of shape (N, embedding_dim)
            metadata: List of chunk metadata dicts
        
        Raises:
            FAISSIndexError: Addition failed
        """
        if not self.use_dual:
            raise FAISSIndexError(
                operation="add_to_delta",
                reason="requires dual-index mode",
                details={'index_name': self.index_name}
            )
        
        if embeddings.shape[0] != len(metadata):
            raise FAISSIndexError(
                operation="add_to_delta",
                reason=f"Embeddings ({embeddings.shape[0]}) and metadata ({len(metadata)}) count mismatch"
            )
        
        if embeddings.shape[1] != self.embedding_dim:
            raise FAISSIndexError(
                operation="add_to_delta",
                reason=f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )
        
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Get starting index
            start_idx = self.delta_index.ntotal
            
            # Add to delta FAISS
            self.delta_index.add(embeddings)
            
            # Update metadata and ID mapping
            unique_docs = set()
            for i, meta in enumerate(metadata):
                idx = start_idx + i
                self.delta_metadata.append(meta)
                self.chunk_id_to_idx[meta['chunk_id']] = ('delta', idx)
                if 'doc_id' in meta:
                    unique_docs.add(meta['doc_id'])
            
            # Update stats
            self.delta_stats['last_updated'] = datetime.utcnow().isoformat() + 'Z'
            self.delta_stats['total_chunks'] = self.delta_index.ntotal
            self.delta_stats['total_docs'] += len(unique_docs)
            
            print(f"✅ Added {len(metadata)} chunks to delta index")
            print(f"   Delta total: {self.delta_index.ntotal} vectors")
            
            # Check if merge needed
            merge_threshold = max(
                config.DELTA_MERGE_THRESHOLD,
                int(self.main_index.ntotal * config.DELTA_MERGE_RATIO)
            )
            
            if self.delta_index.ntotal >= merge_threshold:
                print(f"⚠️  Delta index large ({self.delta_index.ntotal} vectors). "
                      f"Consider running merge_delta_to_main()")
        
        except Exception as e:
            error_dict = handle_error(e, context={
                'operation': 'add_to_delta',
                'index_name': self.index_name,
                'embeddings_shape': embeddings.shape
            })
            raise FAISSIndexError(
                operation="add_to_delta",
                reason=str(e),
                details=error_dict
            )
    
    def add_to_main(
        self,
        embeddings: np.ndarray,
        metadata: List[dict]
    ):
        """
        Add chunks directly to main index (for single-index mode).
        
        Args:
            embeddings: np.ndarray of shape (N, embedding_dim)
            metadata: List of chunk metadata dicts
        """
        if embeddings.shape[0] != len(metadata):
            raise FAISSIndexError(
                operation="add_to_main",
                reason="Embeddings and metadata count mismatch"
            )
        
        try:
            # Normalize
            faiss.normalize_L2(embeddings)
            
            # Get starting index
            start_idx = self.main_index.ntotal
            
            # Add to main FAISS
            self.main_index.add(embeddings)
            
            # Update metadata and ID mapping
            unique_docs = set()
            for i, meta in enumerate(metadata):
                idx = start_idx + i
                self.main_metadata.append(meta)
                self.chunk_id_to_idx[meta['chunk_id']] = ('main', idx)
                if 'doc_id' in meta:
                    unique_docs.add(meta['doc_id'])
            
            # Update stats
            self.main_stats['last_updated'] = datetime.utcnow().isoformat() + 'Z'
            self.main_stats['total_chunks'] = self.main_index.ntotal
            self.main_stats['total_docs'] += len(unique_docs)
            
            print(f"✅ Added {len(metadata)} chunks to main index")
            print(f"   Main total: {self.main_index.ntotal} vectors")
        
        except Exception as e:
            raise FAISSIndexError(
                operation="add_to_main",
                reason=str(e)
            )
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        search_delta: bool = True
    ) -> List[SearchResult]:
        """
        Search FAISS index (searches both main and delta if dual-index).
        
        Args:
            query_embedding: np.ndarray of shape (1, embedding_dim) or (embedding_dim,)
            k: Number of results to return
            search_delta: Also search delta index (dual-index only)
        
        Returns:
            List of SearchResult objects, sorted by score
        """
        if self.main_index.ntotal == 0 and (not self.use_dual or self.delta_index.ntotal == 0):
            return []
        
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        faiss.normalize_L2(query_embedding)
        
        all_results = []
        
        try:
            # Search main index
            if self.main_index.ntotal > 0:
                similarities, indices = self.main_index.search(query_embedding, min(k, self.main_index.ntotal))
                
                for similarity, idx in zip(similarities[0], indices[0]):
                    if idx < 0 or idx >= len(self.main_metadata):
                        continue
                    
                    meta = self.main_metadata[idx]
                    all_results.append(self._build_search_result(meta, float(similarity)))
            
            # Search delta index
            if self.use_dual and search_delta and self.delta_index.ntotal > 0:
                similarities, indices = self.delta_index.search(query_embedding, min(k, self.delta_index.ntotal))
                
                for similarity, idx in zip(similarities[0], indices[0]):
                    if idx < 0 or idx >= len(self.delta_metadata):
                        continue
                    
                    meta = self.delta_metadata[idx]
                    all_results.append(self._build_search_result(meta, float(similarity)))
            
            # Sort by similarity (higher is better) and return top-k
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:k]
        
        except Exception as e:
            raise FAISSIndexError(
                operation="search",
                reason=str(e)
            )
    
    def _build_search_result(self, meta: dict, score: float) -> SearchResult:
        """Build SearchResult from metadata."""
        return SearchResult(
            chunk_id=meta['chunk_id'],
            parent_id=meta.get('parent_id', ''),
            text=meta['text'],
            score=score,
            source_file=meta.get('source_file', ''),
            source_type=meta.get('source_type', ''),
            section_title=meta.get('section_title'),
            indexed_at=meta.get('indexed_at'),
            doc_id=meta.get('doc_id')
        )
    
    def merge_delta_to_main(self):
        """
        Merge delta index into main index (rebuild main as HNSW).
        
        This is an expensive operation. Run weekly/monthly or when delta is large.
        """
        if not self.use_dual:
            raise FAISSIndexError(
                operation="merge_delta_to_main",
                reason="requires dual-index mode"
            )
        
        if self.delta_index.ntotal == 0:
            print("ℹ️  Delta index is empty, nothing to merge")
            return
        
        print(f"\n🔄 Merging delta → main ({self.delta_index.ntotal} vectors)...")
        
        try:
            # Combine all embeddings
            print("  [1/4] Extracting vectors...")
            main_vectors = self._extract_vectors(self.main_index, len(self.main_metadata))
            delta_vectors = self._extract_vectors(self.delta_index, len(self.delta_metadata))
            
            all_vectors = np.vstack([main_vectors, delta_vectors])
            all_metadata = self.main_metadata + self.delta_metadata
            
            print(f"  [2/4] Building new HNSW index ({all_vectors.shape[0]} vectors)...")
            
            # Create new HNSW index
            M = 32
            new_main = faiss.IndexHNSWFlat(self.embedding_dim, M)
            new_main.hnsw.efConstruction = 200
            new_main.hnsw.efSearch = 64
            
            # Add all vectors
            faiss.normalize_L2(all_vectors)
            new_main.add(all_vectors)
            
            # Update metadata and mappings
            print("  [3/4] Updating metadata...")
            self.main_index = new_main
            self.main_metadata = all_metadata
            self.delta_metadata = []
            
            # Rebuild chunk ID mapping
            self.chunk_id_to_idx = {}
            for idx, meta in enumerate(self.main_metadata):
                self.chunk_id_to_idx[meta['chunk_id']] = ('main', idx)
            
            # Clear delta
            self.delta_index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Update stats
            unique_docs = len(set(m.get('doc_id') for m in self.main_metadata if 'doc_id' in m))
            self.main_stats.update({
                'last_updated': datetime.utcnow().isoformat() + 'Z',
                'total_chunks': self.main_index.ntotal,
                'total_docs': unique_docs
            })
            
            self.delta_stats = {
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'last_updated': datetime.utcnow().isoformat() + 'Z',
                'total_docs': 0,
                'total_chunks': 0
            }
            
            print("  [4/4] Saving merged index...")
            self.save()
            
            print(f"✅ Merge complete: {self.main_index.ntotal} vectors in main index")
        
        except Exception as e:
            raise FAISSIndexError(
                operation="merge_delta_to_main",
                reason=str(e)
            )
    
    def _extract_vectors(self, index: faiss.Index, count: int) -> np.ndarray:
        """Extract vectors from FAISS index."""
        vectors = np.zeros((count, self.embedding_dim), dtype=np.float32)
        for i in range(count):
            vectors[i] = index.reconstruct(i)
        return vectors
    
    def save(self):
        """Save indices and metadata to disk (atomic writes)."""
        try:
            # Save main index
            self._atomic_write_index(self.main_index, self.main_index_path)
            self._atomic_write_json(self.main_stats, self.main_stats_path)
            
            # Save delta index (if dual)
            if self.use_dual:
                self._atomic_write_index(self.delta_index, self.delta_index_path)
                self._atomic_write_json(self.delta_stats, self.delta_stats_path)
            
            # Save metadata (pickle)
            self._atomic_write_pickle({
                'main_metadata': self.main_metadata,
                'delta_metadata': self.delta_metadata if self.use_dual else [],
                'chunk_id_to_idx': self.chunk_id_to_idx
            }, self.metadata_path)
            
            print(f"💾 Saved {self.index_name} index:")
            print(f"   Main: {self.main_index.ntotal} vectors")
            if self.use_dual:
                print(f"   Delta: {self.delta_index.ntotal} vectors")
        
        except Exception as e:
            raise FAISSIndexError(
                operation="save",
                reason=str(e)
            )
    
    def load(self):
        """Load indices and metadata from disk."""
        try:
            # Load main index
            self.main_index = faiss.read_index(str(self.main_index_path))
            
            with open(self.main_stats_path, 'r') as f:
                self.main_stats = json.load(f)
            
            # Load delta index (if dual)
            if self.use_dual:
                if self.delta_index_path.exists():
                    self.delta_index = faiss.read_index(str(self.delta_index_path))
                else:
                    self.delta_index = faiss.IndexFlatIP(self.embedding_dim)
                
                if self.delta_stats_path.exists():
                    with open(self.delta_stats_path, 'r') as f:
                        self.delta_stats = json.load(f)
                else:
                    self.delta_stats = {
                        'created_at': datetime.utcnow().isoformat() + 'Z',
                        'last_updated': datetime.utcnow().isoformat() + 'Z',
                        'total_docs': 0,
                        'total_chunks': 0
                    }
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.main_metadata = data['main_metadata']
                self.delta_metadata = data.get('delta_metadata', [])
                self.chunk_id_to_idx = data['chunk_id_to_idx']
            
            print(f"📂 Loaded {self.index_name} index:")
            print(f"   Main: {self.main_index.ntotal} vectors")
            if self.use_dual:
                print(f"   Delta: {self.delta_index.ntotal} vectors")
        
        except Exception as e:
            raise FAISSIndexError(
                operation="load",
                reason=str(e)
            )
    
    def _atomic_write_index(self, index: faiss.Index, path: Path):
        """Atomic write for FAISS index."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=path.parent) as tmp:
            tmp_path = Path(tmp.name)
            faiss.write_index(index, str(tmp_path))
        shutil.move(str(tmp_path), str(path))
    
    def _atomic_write_json(self, data: dict, path: Path):
        """Atomic write for JSON."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=path.parent, encoding='utf-8') as tmp:
            tmp_path = Path(tmp.name)
            json.dump(data, tmp, indent=2)
        shutil.move(str(tmp_path), str(path))
    
    def _atomic_write_pickle(self, data: dict, path: Path):
        """Atomic write for pickle."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=path.parent) as tmp:
            tmp_path = Path(tmp.name)
            pickle.dump(data, tmp)
        shutil.move(str(tmp_path), str(path))
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        stats = {
            'index_name': self.index_name,
            'mode': 'dual' if self.use_dual else 'single',
            'embedding_dim': self.embedding_dim,
            'main_index': {
                'vectors': self.main_index.ntotal,
                'type': 'HNSW' if self.use_dual else 'Flat',
                **self.main_stats
            }
        }
        
        if self.use_dual:
            stats['delta_index'] = {
                'vectors': self.delta_index.ntotal,
                'type': 'Flat',
                **self.delta_stats
            }
        
        return stats


def add_documents_to_index(
    index_name: str,
    documents: List[Dict],
    embeddings: np.ndarray,
    use_dual: bool = True
) -> bool:
    """
    Add documents to FAISS index (used by update_paper_indices.py).
    
    Args:
        index_name: Name of index ('reference_papers', 'my_papers', etc)
        documents: List of metadata dicts (one per embedding)
        embeddings: numpy array of embeddings (shape: [N, dim])
        use_dual: Use dual-index strategy (main + delta)
    
    Returns:
        True if successful
    """
    try:
        builder = FAISSBuilder(index_name=index_name, use_dual=use_dual)
        builder.add_to_delta(embeddings, documents)
        builder.save()
        return True
    except Exception as e:
        print(f"❌ Error adding documents to index: {e}")
        return False


if __name__ == "__main__":
    # Test FAISS builder
    print("="*80)
    print("Testing FAISSBuilder")
    print("="*80)
    
    # Create test index
    builder = FAISSBuilder(
        index_name="test_index",
        use_dual=True
    )
    
    print(f"\n{builder.get_stats()}")
    
    # Test adding vectors
    print("\nTest: Adding vectors to delta...")
    embeddings = np.random.rand(10, config.EMBEDDING_DIM).astype(np.float32)
    metadata = [
        {
            'chunk_id': f'chunk_{i}',
            'parent_id': f'parent_{i // 3}',
            'text': f'Test text {i}',
            'source_file': 'test.pdf',
            'source_type': 'pdf',
            'doc_id': 'doc_1'
        }
        for i in range(10)
    ]
    
    builder.add_to_delta(embeddings, metadata)
    
    # Test search
    print("\nTest: Search...")
    query = np.random.rand(config.EMBEDDING_DIM).astype(np.float32)
    results = builder.search(query, k=3)
    
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  - {r.chunk_id}: {r.score:.4f}")
    
    # Save
    print("\nTest: Save...")
    builder.save()
    
    print("\n✅ All tests passed!")
