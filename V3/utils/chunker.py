"""
chunker.py

Hierarchical text chunking for MRA_v3.

Strategy:
- Parent chunks: 2000 tokens (full context for user display)
- Child chunks: 500 tokens (precise search in FAISS)
- 100 token overlap between children
- Links children to parents via metadata

Adapted from:
- TwistedPair/V4/document_processor.py (chunk_text_hierarchical)
- PDF2TextConversion/MRA_2chunk_md.py (tokenization)

Usage:
    from utils.chunker import Chunker
    
    chunker = Chunker()
    result = chunker.chunk_hierarchical(markdown_text)
    
    # Result contains:
    # - parents: List of parent chunks (2000 tokens each)
    # - children: List of child chunks (500 tokens each, linked to parents)
"""
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import tiktoken

from errors import ChunkingError, handle_error


@dataclass
class ChildChunk:
    """
    Child chunk for FAISS indexing.
    
    Small chunks (500 tokens) for precise semantic search.
    """
    chunk_id: str
    parent_id: str
    text: str
    tokens: int
    child_position: int  # Position within parent (0-indexed)
    start_offset: int    # Character offset in original text
    end_offset: int      # Character offset in original text


@dataclass
class ParentChunk:
    """
    Parent chunk for context retrieval.
    
    Large chunks (2000 tokens) providing full context to LLM.
    """
    parent_id: str
    text: str
    tokens: int
    child_ids: List[str]
    start_offset: int    # Character offset in original text
    end_offset: int      # Character offset in original text


@dataclass
class ChunkingResult:
    """Result of hierarchical chunking operation."""
    parents: List[ParentChunk]
    children: List[ChildChunk]
    total_parents: int
    total_children: int
    original_tokens: int
    chunked_at: str


class Chunker:
    """
    Hierarchical text chunker with parent/child strategy.
    
    Features:
    - Token-aware chunking using tiktoken
    - Parent chunks: 2000 tokens (no overlap)
    - Child chunks: 500 tokens (100 token overlap)
    - Preserves context at boundaries
    - Links children to parents
    """
    
    def __init__(
        self,
        parent_size: int = 2000,
        child_size: int = 500,
        overlap: int = 100,
        tokenizer_model: str = "cl100k_base"
    ):
        """
        Initialize chunker.
        
        Args:
            parent_size: Parent chunk size in tokens (default: 2000)
            child_size: Child chunk size in tokens (default: 500)
            overlap: Token overlap between children (default: 100)
            tokenizer_model: tiktoken model name (default: cl100k_base)
        """
        if overlap >= child_size:
            raise ValueError(f"Overlap ({overlap}) must be less than child size ({child_size})")
        
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap
        self.tokenizer_model = tokenizer_model
        
        try:
            self.encoding = tiktoken.get_encoding(tokenizer_model)
        except Exception as e:
            raise ChunkingError(reason=f"Failed to load tiktoken encoding: {e}")
    
    def tokenize(self, text: str) -> List[int]:
        """Encode text to tokens."""
        try:
            return self.encoding.encode(text)
        except Exception as e:
            raise ChunkingError(reason=f"Tokenization failed: {e}")
    
    def detokenize(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        try:
            return self.encoding.decode(tokens)
        except Exception as e:
            raise ChunkingError(reason=f"Detokenization failed: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenize(text))
    
    def chunk_hierarchical(
        self,
        text: str,
        doc_id: Optional[str] = None,
        section_title: Optional[str] = None
    ) -> ChunkingResult:
        """
        Create hierarchical chunks: parents (2000 tokens) and children (500 tokens).
        
        Args:
            text: Input text to chunk
            doc_id: Optional document ID (for metadata)
            section_title: Optional section title (for metadata)
        
        Returns:
            ChunkingResult with parents and children
        
        Raises:
            ChunkingError: Chunking failed
        """
        try:
            # Tokenize entire text
            tokens = self.tokenize(text)
            total_tokens = len(tokens)
            
            if total_tokens == 0:
                return ChunkingResult(
                    parents=[],
                    children=[],
                    total_parents=0,
                    total_children=0,
                    original_tokens=0,
                    chunked_at=datetime.utcnow().isoformat() + "Z"
                )
            
            # Step 1: Create parent chunks (no overlap)
            parents = []
            parent_start = 0
            char_offset = 0
            
            while parent_start < total_tokens:
                parent_end = min(parent_start + self.parent_size, total_tokens)
                parent_tokens = tokens[parent_start:parent_end]
                parent_text = self.detokenize(parent_tokens)
                
                parent_id = str(uuid.uuid4())
                
                parent_chunk = ParentChunk(
                    parent_id=parent_id,
                    text=parent_text.strip(),
                    tokens=len(parent_tokens),
                    child_ids=[],  # Will populate after creating children
                    start_offset=char_offset,
                    end_offset=char_offset + len(parent_text)
                )
                
                parents.append(parent_chunk)
                parent_start = parent_end
                char_offset += len(parent_text)
            
            # Step 2: Create child chunks within each parent (with overlap)
            all_children = []
            
            for parent_idx, parent in enumerate(parents):
                parent_tokens = self.tokenize(parent.text)
                step = self.child_size - self.overlap
                
                child_start = 0
                child_position = 0
                
                while child_start < len(parent_tokens):
                    child_end = min(child_start + self.child_size, len(parent_tokens))
                    child_tokens = parent_tokens[child_start:child_end]
                    
                    # Skip tiny chunks at the end
                    if len(child_tokens) < 50:
                        break
                    
                    child_text = self.detokenize(child_tokens)
                    child_id = str(uuid.uuid4())
                    
                    child_chunk = ChildChunk(
                        chunk_id=child_id,
                        parent_id=parent.parent_id,
                        text=child_text.strip(),
                        tokens=len(child_tokens),
                        child_position=child_position,
                        start_offset=parent.start_offset + child_start,
                        end_offset=parent.start_offset + child_end
                    )
                    
                    all_children.append(child_chunk)
                    parent.child_ids.append(child_id)
                    
                    child_position += 1
                    child_start += step
                    
                    # Break if we've reached the end
                    if child_end >= len(parent_tokens):
                        break
            
            return ChunkingResult(
                parents=parents,
                children=all_children,
                total_parents=len(parents),
                total_children=len(all_children),
                original_tokens=total_tokens,
                chunked_at=datetime.utcnow().isoformat() + "Z"
            )
        
        except Exception as e:
            error_dict = handle_error(e, context={
                'operation': 'chunk_hierarchical',
                'text_length': len(text),
                'parent_size': self.parent_size,
                'child_size': self.child_size
            })
            raise ChunkingError(
                reason=f"Hierarchical chunking failed: {e}",
                details=error_dict
            )
    
    def chunk_simple(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[Dict]:
        """
        Simple chunking without parent/child hierarchy.
        
        Args:
            text: Input text to chunk
            chunk_size: Chunk size in tokens (default: child_size)
            overlap: Token overlap (default: self.overlap)
        
        Returns:
            List of chunk dictionaries
        """
        chunk_size = chunk_size or self.child_size
        overlap = overlap or self.overlap
        
        tokens = self.tokenize(text)
        total_tokens = len(tokens)
        
        chunks = []
        step = chunk_size - overlap
        start = 0
        
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            
            if len(chunk_tokens) < 50:  # Skip tiny chunks
                break
            
            chunk_text = self.detokenize(chunk_tokens)
            
            chunks.append({
                'chunk_id': str(uuid.uuid4()),
                'text': chunk_text.strip(),
                'tokens': len(chunk_tokens),
                'start': start,
                'end': end
            })
            
            start += step
            
            if end >= total_tokens:
                break
        
        return chunks
    
    def to_dict(self, result: ChunkingResult) -> Dict:
        """Convert ChunkingResult to dictionary."""
        return {
            'parents': [asdict(p) for p in result.parents],
            'children': [asdict(c) for c in result.children],
            'total_parents': result.total_parents,
            'total_children': result.total_children,
            'original_tokens': result.original_tokens,
            'chunked_at': result.chunked_at
        }
    
    def summary(self, result: ChunkingResult) -> str:
        """Generate human-readable summary of chunking result."""
        avg_children_per_parent = (
            result.total_children / result.total_parents 
            if result.total_parents > 0 else 0
        )
        
        return (
            f"Chunking Summary:\n"
            f"  Original tokens: {result.original_tokens:,}\n"
            f"  Parent chunks: {result.total_parents} "
            f"(~{self.parent_size} tokens each)\n"
            f"  Child chunks: {result.total_children} "
            f"(~{self.child_size} tokens each)\n"
            f"  Avg children per parent: {avg_children_per_parent:.1f}\n"
            f"  Overlap: {self.overlap} tokens\n"
            f"  Chunked at: {result.chunked_at}"
        )


# Convenience functions
def chunk_text_hierarchical(
    text: str,
    parent_size: int = 2000,
    child_size: int = 500,
    overlap: int = 100
) -> Tuple[List[str], List[List[str]]]:
    """
    Convenience function: Create hierarchical chunks.
    
    Returns:
        (parent_chunks, children_per_parent)
        - parent_chunks: List of parent chunk texts
        - children_per_parent: List of lists, each containing children for that parent
    """
    chunker = Chunker(
        parent_size=parent_size,
        child_size=child_size,
        overlap=overlap
    )
    
    result = chunker.chunk_hierarchical(text)
    
    # Group children by parent
    children_by_parent = {}
    for child in result.children:
        if child.parent_id not in children_by_parent:
            children_by_parent[child.parent_id] = []
        children_by_parent[child.parent_id].append(child.text)
    
    # Build output
    parent_texts = [p.text for p in result.parents]
    children_per_parent = [
        children_by_parent.get(p.parent_id, []) 
        for p in result.parents
    ]
    
    return parent_texts, children_per_parent


if __name__ == "__main__":
    # Test chunking
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <text_file_or_string>")
        print("\nExample test:")
        print('  python chunker.py "Test text here..."')
        print("  python chunker.py document.txt")
        sys.exit(1)
    
    # Load text
    arg = sys.argv[1]
    if arg.endswith('.txt') or arg.endswith('.md'):
        with open(arg, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded {len(text)} characters from {arg}")
    else:
        text = arg
    
    # Chunk
    chunker = Chunker()
    result = chunker.chunk_hierarchical(text)
    
    # Print summary
    print("\n" + "="*80)
    print(chunker.summary(result))
    print("="*80)
    
    # Show first parent and its children
    if result.parents:
        print("\nFirst Parent Chunk:")
        print("-" * 80)
        first_parent = result.parents[0]
        print(f"ID: {first_parent.parent_id}")
        print(f"Tokens: {first_parent.tokens}")
        print(f"Text preview: {first_parent.text[:200]}...")
        
        print(f"\nChildren ({len(first_parent.child_ids)}):")
        for child in result.children:
            if child.parent_id == first_parent.parent_id:
                print(f"  - Child {child.child_position}: "
                      f"{child.tokens} tokens, "
                      f"'{child.text[:50]}...'")
