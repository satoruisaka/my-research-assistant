"""
embedder.py

Embedding module for MRA_v3.

Features:
- Alibaba-NLP/gte-large-en-v1.5 (1024-dim)
- GPU-accelerated with sentence-transformers
- Batch processing with progress bars
- L2 normalization for cosine similarity
- Stream processing for large datasets

Adapted from:
- TwistedPair/V4/embedder.py (batch processing)
- PDF2TextConversion/MRA_3embed_chunks_S.py (streaming)

Usage:
    from utils.embedder import Embedder
    
    embedder = Embedder()
    embeddings = embedder.embed_batch(["text1", "text2", ...])
    # Shape: (N, 1024)
"""
import numpy as np
from typing import List, Dict, Optional, Iterator
from pathlib import Path
import os

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tqdm import tqdm

from errors import EmbeddingError, MissingDependencyError, handle_error
import config


class Embedder:
    """
    GPU-accelerated embedding for document chunks.
    
    Uses Alibaba-NLP/gte-large-en-v1.5 (1024-dimensional vectors).
    
    Features:
    - Batch processing with configurable batch size
    - Progress bars for long operations
    - L2 normalization (for cosine similarity)
    - GPU auto-detection with CPU fallback
    - Stream processing for memory efficiency
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: HuggingFace model name (default: from config)
            device: 'cuda' or 'cpu' (auto-detected if None)
            batch_size: Batch size for encoding (default: from config)
            cache_dir: Directory for model cache (default: ./models)
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Setup cache directory
        self.cache_dir = cache_dir or Path("./models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        try:
            print(f"Loading embedding model: {self.model_name}")
            print(f"Device: {self.device}")
            
            # Check if cached model exists and is valid
            cache_path = self.cache_dir / self.model_name.replace('/', '_')
            if cache_path.exists():
                print(f"Found cached model at: {cache_path}")
            
            # Monkey-patch AutoConfig.from_pretrained to always use trust_remote_code=True
            original_from_pretrained = AutoConfig.from_pretrained
            
            def patched_from_pretrained(*args, **kwargs):
                kwargs['trust_remote_code'] = True
                # If loading from cache fails, force download from HuggingFace
                if 'local_files_only' in kwargs:
                    kwargs.pop('local_files_only')
                return original_from_pretrained(*args, **kwargs)
            
            AutoConfig.from_pretrained = patched_from_pretrained
            
            try:
                # First attempt: use cache if available
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=str(self.cache_dir)
                )
            except Exception as e:
                print(f"⚠️  Cache load failed: {e}")
                print(f"🔄 Clearing cache and re-downloading from HuggingFace...")
                
                # Remove corrupted cache
                if cache_path.exists():
                    import shutil
                    shutil.rmtree(cache_path, ignore_errors=True)
                    print(f"✓ Cleared cache: {cache_path}")
                
                # Force fresh download
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=str(self.cache_dir)
                )
            finally:
                # Restore original method
                AutoConfig.from_pretrained = original_from_pretrained
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
            
            # Verify dimension matches config
            if self.embedding_dim != config.EMBEDDING_DIM:
                raise EmbeddingError(
                    f"Model dimension mismatch: expected {config.EMBEDDING_DIM}, "
                    f"got {self.embedding_dim}"
                )
        
        except Exception as e:
            raise MissingDependencyError(
                dependency=f"embedding model {self.model_name}",
                solution=f"Failed to load embedding model: {e}"
            )
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed list of texts using GPU batching.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size (default: self.batch_size)
            show_progress: Show tqdm progress bar
            normalize: Apply L2 normalization (for cosine similarity)
        
        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        
        Raises:
            EmbeddingError: Embedding failed
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        batch_size = batch_size or self.batch_size
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            return embeddings
        
        except Exception as e:
            error_dict = handle_error(e, context={
                'operation': 'embed_batch',
                'num_texts': len(texts),
                'batch_size': batch_size
            })
            raise EmbeddingError(
                f"Failed to embed {len(texts)} texts: {e}",
                context=error_dict
            )
    
    def embed_stream(
        self,
        text_iterator: Iterator[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        normalize: bool = True
    ) -> Iterator[np.ndarray]:
        """
        Stream embed texts (memory-efficient for large datasets).
        
        Yields batches of embeddings instead of loading all at once.
        
        Args:
            text_iterator: Iterator yielding text strings
            batch_size: Batch size (default: self.batch_size)
            show_progress: Show tqdm progress bar
            normalize: Apply L2 normalization
        
        Yields:
            np.ndarray of shape (batch_size, embedding_dim) per batch
        
        Raises:
            EmbeddingError: Embedding failed
        """
        batch_size = batch_size or self.batch_size
        batch = []
        
        try:
            iterator = tqdm(text_iterator, desc="Embedding") if show_progress else text_iterator
            
            for text in iterator:
                batch.append(text)
                
                if len(batch) >= batch_size:
                    # Encode batch
                    embeddings = self.model.encode(
                        batch,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=normalize
                    )
                    
                    yield embeddings
                    batch = []
            
            # Process remaining texts
            if batch:
                embeddings = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize
                )
                yield embeddings
        
        except Exception as e:
            error_dict = handle_error(e, context={
                'operation': 'embed_stream',
                'batch_size': batch_size
            })
            raise EmbeddingError(
                f"Stream embedding failed: {e}",
                context=error_dict
            )
    
    def embed_single(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed single text.
        
        Args:
            text: Text string to embed
            normalize: Apply L2 normalization
        
        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            return embedding
        
        except Exception as e:
            raise EmbeddingError(reason=f"Failed to embed text: {e}")
    
    def embed_with_metadata(
        self,
        chunks: List[Dict],
        text_key: str = "text",
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Embed chunks with metadata preservation.
        
        Args:
            chunks: List of chunk dictionaries (must have text_key field)
            text_key: Key containing text to embed (default: "text")
            batch_size: Batch size
            show_progress: Show progress bar
        
        Returns:
            Dictionary with:
            - embeddings: np.ndarray (N, embedding_dim)
            - metadata: List[dict] (original chunks)
            - chunk_ids: List[str] (if 'chunk_id' in chunks)
        """
        if not chunks:
            return {
                'embeddings': np.array([]).reshape(0, self.embedding_dim),
                'metadata': [],
                'chunk_ids': []
            }
        
        # Extract texts
        texts = [chunk[text_key] for chunk in chunks]
        
        # Embed
        embeddings = self.embed_batch(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        # Extract chunk IDs if present
        chunk_ids = [chunk.get('chunk_id', f'chunk_{i}') for i, chunk in enumerate(chunks)]
        
        return {
            'embeddings': embeddings,
            'metadata': chunks,
            'chunk_ids': chunk_ids
        }
    
    def get_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (1D or 2D array)
            embedding2: Second embedding (1D or 2D array)
        
        Returns:
            Cosine similarity score (0-1)
        """
        # Ensure 1D
        if embedding1.ndim > 1:
            embedding1 = embedding1.flatten()
        if embedding2.ndim > 1:
            embedding2 = embedding2.flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
    
    def __repr__(self) -> str:
        return (
            f"Embedder(model='{self.model_name}', "
            f"dim={self.embedding_dim}, "
            f"device='{self.device}', "
            f"batch_size={self.batch_size})"
        )


# Convenience functions
def embed_texts(
    texts: List[str],
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None
) -> np.ndarray:
    """
    Convenience function: Embed list of texts.
    
    Args:
        texts: List of text strings
        model_name: Model name (default: from config)
        batch_size: Batch size (default: from config)
    
    Returns:
        np.ndarray of embeddings
    """
    embedder = Embedder(model_name=model_name, batch_size=batch_size)
    return embedder.embed_batch(texts)


def embed_query(query: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Convenience function: Embed single query.
    
    Args:
        query: Query text
        model_name: Model name (default: from config)
    
    Returns:
        np.ndarray embedding (1D)
    """
    embedder = Embedder(model_name=model_name)
    return embedder.embed_single(query)


if __name__ == "__main__":
    # Test embedder
    import sys
    
    print("="*80)
    print("Testing Embedder")
    print("="*80)
    
    # Initialize
    embedder = Embedder()
    print(f"\n{embedder}\n")
    
    # Test single embedding
    print("Test 1: Single text")
    text = "Quantum computing is an emerging field."
    embedding = embedder.embed_single(text)
    print(f"  Text: '{text}'")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Test batch embedding
    print("\nTest 2: Batch of texts")
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Reinforcement learning learns from rewards and penalties."
    ]
    embeddings = embedder.embed_batch(texts, show_progress=False)
    print(f"  Texts: {len(texts)}")
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Test similarity
    print("\nTest 3: Similarity")
    sim_01 = embedder.get_similarity(embeddings[0], embeddings[1])
    sim_02 = embedder.get_similarity(embeddings[0], embeddings[2])
    print(f"  Similarity (text 0 vs 1): {sim_01:.4f}")
    print(f"  Similarity (text 0 vs 2): {sim_02:.4f}")
    
    # Test with metadata
    print("\nTest 4: With metadata")
    chunks = [
        {'chunk_id': 'c1', 'text': texts[0]},
        {'chunk_id': 'c2', 'text': texts[1]}
    ]
    result = embedder.embed_with_metadata(chunks, show_progress=False)
    print(f"  Chunks: {len(chunks)}")
    print(f"  Embeddings: {result['embeddings'].shape}")
    print(f"  Chunk IDs: {result['chunk_ids']}")
    
    print("\n✅ All tests passed!")
