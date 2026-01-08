"""
retrieval_manager.py - Unified FAISS Search Manager

Provides unified search across 4 FAISS indices with result merging,
score normalization, and parent chunk retrieval.

Indices:
- reference_papers: Academic papers from MyReferences (main corpus)
- my_papers: User's authored papers from MyPapers
- sessions: Past chat sessions (auto-indexed)
- web_cache: Web search results (auto-indexed)

Architecture:
1. Query embedding via Embedder
2. Parallel search across selected indices
3. Score normalization and merging
4. Parent chunk retrieval from metadata
5. Sorting by normalized score
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

from utils.embedder import Embedder
from utils.faiss_builder import FAISSBuilder, SearchResult


@dataclass
class RetrievalResult:
    """Unified retrieval result across all indices."""
    chunk_id: str
    parent_id: str
    parent_text: str
    child_text: str
    score: float  # Normalized 0-1
    source: str  # 'reference_papers', 'my_papers', 'sessions', 'web_cache'
    doc_id: str
    filename: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


@dataclass
class SearchScope:
    """Defines which indices to search."""
    reference_papers: bool = True
    my_papers: bool = True
    sessions: bool = False
    web_cache: bool = False
    
    def get_active_indices(self) -> List[str]:
        """Return list of index names where flag is True."""
        return [name for name, enabled in asdict(self).items() if enabled]


class RetrievalManager:
    """
    Manages unified search across multiple FAISS indices.
    
    Features:
    - Parallel index search
    - Score normalization (min-max per index)
    - Result merging and deduplication
    - Parent chunk retrieval
    - Metadata filtering
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        embedder: Optional[Embedder] = None,
        verbose: bool = False
    ):
        """
        Initialize retrieval manager.
        
        Args:
            data_dir: Root data directory containing index subdirectories
            embedder: Embedder instance (creates new if None)
            verbose: Enable debug logging
        """
        self.data_dir = Path(data_dir)
        self._embedder = embedder  # Store provided embedder, or None for lazy init
        self._embedder_config = None  # Store config if embedder was provided
        self.verbose = verbose
        
        # Use faiss_indices directory (from staged pipeline)
        from config import FAISS_DIR
        self.faiss_dir = Path(FAISS_DIR)
        
        # Index names (map to FAISSBuilder index_name parameter)
        self.index_names = ['reference_papers', 'my_papers', 'sessions', 'web_cache']
        
        # Load indices (lazy loading - only when needed)
        self.indices: Dict[str, FAISSBuilder] = {}
        self.metadata_cache: Dict[str, Dict] = {}  # index_name -> metadata dict
        
        self._log("RetrievalManager initialized (embedder lazy-loaded)")
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[RetrievalManager] {message}")
    
    @property
    def embedder(self) -> Embedder:
        """Lazy-load embedder on first access."""
        if self._embedder is None:
            self._log("Loading embedder on first use...")
            self._embedder = Embedder()
        return self._embedder
    
    def unload_embedder(self) -> None:
        """
        Unload embedding model from GPU to free memory.
        Useful when MRA is running alongside other GPU services (Ollama, TwistedPic).
        """
        if self._embedder is not None:
            self._embedder.unload_from_gpu()
    
    def reload_embedder(self) -> None:
        """
        Reload embedding model to GPU (automatic on next use).
        """
        if self._embedder is not None:
            self._embedder.reload_to_gpu()
        else:
            self._log("Embedder not yet loaded, will load on first use")
    
    def is_embedder_loaded(self) -> bool:
        """Check if embedder is currently loaded in memory."""
        return self._embedder is not None and hasattr(self._embedder, 'model')
    
    def _load_index(self, index_name: str) -> Optional[FAISSBuilder]:
        """
        Load FAISS index if not already loaded.
        
        Args:
            index_name: Name of index to load
            
        Returns:
            FAISSBuilder instance or None if not found
        """
        if index_name in self.indices:
            return self.indices[index_name]
        
        # Determine if dual-index mode
        use_dual = index_name in ['reference_papers', 'sessions', 'web_cache']
        
        try:
            # Use FAISSBuilder with index_name (matches staged pipeline)
            builder = FAISSBuilder(
                index_name=index_name,
                use_dual=use_dual
            )
            
            # Try to load existing index
            try:
                builder.load()
                self._log(f"Loaded {index_name}: Main={builder.main_index.ntotal} vectors")
                if use_dual:
                    self._log(f"  Delta={builder.delta_index.ntotal} vectors")
            except Exception as e:
                self._log(f"Index not found or failed to load: {index_name} ({e})")
                return None
            
            self.indices[index_name] = builder
            return builder
            
        except Exception as e:
            self._log(f"Error loading {index_name}: {e}")
            return None
    
    def _normalize_scores(
        self,
        results: List[SearchResult],
        method: str = 'minmax'
    ) -> List[SearchResult]:
        """
        Normalize similarity scores to 0-1 range.
        
        Args:
            results: List of search results
            method: 'minmax' or 'sigmoid'
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
        
        scores = np.array([r.score for r in results])
        
        self._log(f"Normalizing {len(scores)} scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        
        if method == 'minmax':
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                normalized = (scores - min_score) / (max_score - min_score)
            else:
                normalized = np.ones_like(scores)
        
        elif method == 'sigmoid':
            # Sigmoid: 1 / (1 + exp(-x))
            # Shift so mean=0, then apply sigmoid
            mean_score = scores.mean()
            std_score = scores.std() + 1e-10
            z_scores = (scores - mean_score) / std_score
            normalized = 1.0 / (1.0 + np.exp(-z_scores))
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Update scores
        for i, result in enumerate(results):
            result.score = float(normalized[i])
        
        return results
    
    def _get_parent_chunk(
        self,
        child_chunk_id: str,
        index_name: str
    ) -> Optional[Dict]:
        """
        Retrieve parent chunk from metadata.
        
        Args:
            child_chunk_id: Child chunk ID
            index_name: Index to search in
            
        Returns:
            Parent chunk metadata or None
        """
        metadata = self.metadata_cache.get(index_name, {})
        child_meta = metadata.get(child_chunk_id)
        
        if not child_meta:
            return None
        
        parent_id = child_meta.get('parent_id')
        if not parent_id:
            return None
        
        # Find parent in metadata
        for chunk_id, chunk_meta in metadata.items():
            if chunk_meta.get('chunk_id') == parent_id:
                return chunk_meta
        
        return None
    
    def search_index(
        self,
        query_embedding: np.ndarray,
        index_name: str,
        k: int = 10
    ) -> List[SearchResult]:
        """
        Search a single index.
        
        Args:
            query_embedding: Query embedding vector (1024-dim)
            index_name: Name of index to search
            k: Number of results to return
            
        Returns:
            List of search results with normalized scores
        """
        builder = self._load_index(index_name)
        if not builder:
            return []
        
        try:
            results = builder.search(
                query_embedding=query_embedding,
                k=k
            )
            
            # Don't normalize per-index - Inner Product scores are already comparable (0-1 range)
            # Normalizing each index independently makes cross-index comparison meaningless
            
            self._log(f"Searched {index_name}: {len(results)} results")
            return results
            
        except Exception as e:
            self._log(f"Error searching {index_name}: {e}")
            return []
    
    def unified_search(
        self,
        query: str,
        scope: Optional[SearchScope] = None,
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Unified search across multiple indices.
        
        Args:
            query: Search query text
            scope: Which indices to search (default: all)
            k: Total number of results to return
            filters: Optional metadata filters (date_range, source_files, etc.)
            
        Returns:
            List of RetrievalResult sorted by score (highest first)
        """
        if scope is None:
            scope = SearchScope()  # Default: all except sessions/web_cache
        
        active_indices = scope.get_active_indices()
        if not active_indices:
            self._log("No indices selected in scope")
            return []
        
        self._log(f"Searching: {active_indices} for query: '{query[:50]}...'")
        
        # 1. Generate query embedding
        query_embedding = self.embedder.embed_single(query)
        
        # Unload embedder to free GPU memory if configured
        from config import UNLOAD_EMBEDDER_AFTER_USE
        if UNLOAD_EMBEDDER_AFTER_USE:
            self.embedder.unload_from_gpu()
        
        # 2. Search each index in parallel (sequential for now, can parallelize later)
        k_per_index = max(k, 20)  # Get more results per index for better merging
        all_results: List[Tuple[str, SearchResult]] = []
        
        for index_name in active_indices:
            results = self.search_index(
                query_embedding=query_embedding,
                index_name=index_name,
                k=k_per_index
            )
            # Tag results with source index
            for result in results:
                all_results.append((index_name, result))
        
        if not all_results:
            self._log("No results found in any index")
            return []
        
        # 3. Global score normalization (optional - scores already normalized per index)
        # Since we normalized per-index, scores are comparable
        
        # 4. Build RetrievalResult objects with parent chunks
        retrieval_results: List[RetrievalResult] = []
        seen_chunks: Set[str] = set()  # Deduplication
        
        for index_name, search_result in all_results:
            chunk_id = search_result.chunk_id
            
            # Skip duplicates
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            
            # Build result directly from SearchResult (no parent lookup needed, SearchResult has all data)
            result = RetrievalResult(
                chunk_id=chunk_id,
                parent_id=search_result.parent_id,
                parent_text='',  # Parent text not needed for display, child text is sufficient
                child_text=search_result.text,
                score=search_result.score,
                source=index_name,
                doc_id=search_result.doc_id or '',
                filename=search_result.source_file or '',
                metadata={
                    'source_file': search_result.source_file,
                    'source_type': search_result.source_type,
                    'section_title': search_result.section_title,
                    'indexed_at': search_result.indexed_at
                }
            )
            
            retrieval_results.append(result)
        
        # 5. Apply filters
        if filters:
            retrieval_results = self._apply_filters(retrieval_results, filters)
        
        # 6. Sort by score and limit to k
        retrieval_results.sort(key=lambda x: x.score, reverse=True)
        retrieval_results = retrieval_results[:k]
        
        self._log(f"Returning {len(retrieval_results)} unified results")
        return retrieval_results
    
    def _apply_filters(
        self,
        results: List[RetrievalResult],
        filters: Dict
    ) -> List[RetrievalResult]:
        """
        Apply metadata filters to results.
        
        Supported filters:
        - date_range: {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}
        - source_files: List[str] - filter by filename
        - min_score: float - minimum similarity score
        
        Args:
            results: List of retrieval results
            filters: Filter dictionary
            
        Returns:
            Filtered results
        """
        filtered = results
        
        # Date range filter
        if 'date_range' in filters:
            date_range = filters['date_range']
            start = datetime.fromisoformat(date_range.get('start', '1900-01-01'))
            end = datetime.fromisoformat(date_range.get('end', '2100-12-31'))
            
            filtered = [
                r for r in filtered
                if 'timestamp' in r.metadata and
                start <= datetime.fromisoformat(r.metadata['timestamp']) <= end
            ]
        
        # Source files filter
        if 'source_files' in filters:
            allowed_files = set(filters['source_files'])
            filtered = [r for r in filtered if r.filename in allowed_files]
        
        # Minimum score filter
        if 'min_score' in filters:
            min_score = filters['min_score']
            filtered = [r for r in filtered if r.score >= min_score]
        
        return filtered
    
    def get_index_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all indices.
        
        Returns:
            Dict mapping index_name to stats (chunks, docs, size_mb)
        """
        stats = {}
        
        for index_name in self.index_names:
            # Determine index type
            use_dual = index_name in ['reference_papers', 'sessions', 'web_cache']
            
            # Check if index files exist
            main_index_path = self.faiss_dir / f"{index_name}_main.index" if use_dual else self.faiss_dir / f"{index_name}.index"
            
            if not main_index_path.exists():
                stats[index_name] = {
                    'exists': False,
                    'chunks': 0,
                    'docs': 0,
                    'size_mb': 0
                }
                continue
            
            try:
                # Load builder to get stats
                builder = FAISSBuilder(index_name=index_name, use_dual=use_dual)
                builder.load()
                
                # Get counts
                main_chunks = builder.main_index.ntotal
                delta_chunks = builder.delta_index.ntotal if use_dual else 0
                total_chunks = main_chunks + delta_chunks
                
                # Count unique docs from metadata
                all_metadata = builder.main_metadata + (builder.delta_metadata if use_dual else [])
                unique_docs = len(set(m.get('doc_id', '') for m in all_metadata if m.get('doc_id')))
                
                # Calculate size
                total_size = sum(
                    f.stat().st_size
                    for f in self.faiss_dir.glob(f"{index_name}*")
                    if f.is_file()
                )
                size_mb = total_size / (1024 * 1024)
                
                stats[index_name] = {
                    'exists': True,
                    'chunks': total_chunks,
                    'docs': unique_docs,
                    'size_mb': round(size_mb, 2)
                }
                
            except Exception as e:
                self._log(f"Failed to get stats for {index_name}: {e}")
                stats[index_name] = {
                    'exists': False,
                    'chunks': 0,
                    'docs': 0,
                    'size_mb': 0
                }
        
        return stats
    
    def refresh_indices(self):
        """Reload all indices and metadata from disk."""
        self._log("Refreshing all indices...")
        self.indices.clear()
        self.metadata_cache.clear()
        
        for index_name in self.index_names:
            self._load_index(index_name)
        
        self._log("All indices refreshed")


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RetrievalManager')
    parser.add_argument('--query', type=str, default='neural networks',
                       help='Search query')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of results')
    parser.add_argument('--scope', type=str, nargs='+',
                       default=['reference_papers'],
                       help='Indices to search (reference_papers my_papers sessions web_cache)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Build scope
    scope = SearchScope(
        reference_papers='reference_papers' in args.scope,
        my_papers='my_papers' in args.scope,
        sessions='sessions' in args.scope,
        web_cache='web_cache' in args.scope
    )
    
    # Initialize manager
    print(f"Initializing RetrievalManager...")
    manager = RetrievalManager(
        data_dir=args.data_dir,
        verbose=args.verbose
    )
    
    # Get stats
    print("\nIndex Statistics:")
    stats = manager.get_index_stats()
    for index_name, index_stats in stats.items():
        if index_stats['exists']:
            print(f"  {index_name}: {index_stats['chunks']} chunks, "
                  f"{index_stats['docs']} docs, {index_stats['size_mb']} MB")
        else:
            print(f"  {index_name}: NOT FOUND")
    
    # Search
    print(f"\nSearching for: '{args.query}'")
    print(f"Scope: {args.scope}")
    print(f"Top-{args.k} results:\n")
    
    results = manager.unified_search(
        query=args.query,
        scope=scope,
        k=args.k
    )
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.source}] {result.filename}")
        print(f"   Score: {result.score:.4f}")
        print(f"   Parent: {result.parent_text[:100]}...")
        print(f"   Child: {result.child_text[:100]}...")
        print()
    
    print(f"\nTotal results: {len(results)}")
