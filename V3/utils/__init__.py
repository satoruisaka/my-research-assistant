"""
__init__.py

MRA_v3 utils package.

Core utilities for document processing, chunking, embedding, and FAISS indexing.
"""
from .document_processor import DocumentProcessor, convert_pdf_to_markdown, convert_document_to_markdown
from .chunker import Chunker, chunk_text_hierarchical, ChildChunk, ParentChunk, ChunkingResult
from .embedder import Embedder, embed_texts, embed_query
from .faiss_builder import FAISSBuilder, SearchResult

__all__ = [
    # Document processing
    'DocumentProcessor',
    'convert_pdf_to_markdown',
    'convert_document_to_markdown',
    
    # Chunking
    'Chunker',
    'chunk_text_hierarchical',
    'ChildChunk',
    'ParentChunk',
    'ChunkingResult',
    
    # Embedding
    'Embedder',
    'embed_texts',
    'embed_query',
    
    # FAISS
    'FAISSBuilder',
    'SearchResult'
]
