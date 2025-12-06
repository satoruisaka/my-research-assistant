"""
test_pipeline.py

Integration test for MRA_v3 core utilities pipeline.

Tests the complete workflow:
1. Document processing (PDF/DOCX/TXT → Markdown)
2. Hierarchical chunking (2000-token parents, 500-token children)
3. Embedding (Alibaba GTE 1024-dim)
4. FAISS indexing (dual-index architecture)
5. Search and retrieval

Usage:
    python utils/test_pipeline.py
    python utils/test_pipeline.py --sample-doc path/to/document.pdf
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.document_processor import DocumentProcessor
from utils.chunker import Chunker
from utils.embedder import Embedder
from utils.faiss_builder import FAISSBuilder
import config


def test_document_processing(doc_path: Path):
    """Test document → Markdown conversion."""
    print("\n" + "="*80)
    print("TEST 1: Document Processing")
    print("="*80)
    
    processor = DocumentProcessor()
    
    print(f"\nProcessing: {doc_path.name}")
    md_text = processor.convert_to_markdown(doc_path)
    
    # Extract metadata
    metadata = processor.extract_metadata_from_markdown(md_text)
    
    print(f"✅ Conversion successful")
    print(f"   Token count: {metadata['token_count']:,}")
    print(f"   File hash: {metadata['file_hash']}")
    print(f"   Preview: {md_text[:200]}...")
    
    return md_text


def test_chunking(md_text: str):
    """Test hierarchical chunking."""
    print("\n" + "="*80)
    print("TEST 2: Hierarchical Chunking")
    print("="*80)
    
    chunker = Chunker(
        parent_size=config.PARENT_CHUNK_SIZE,
        child_size=config.CHILD_CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP
    )
    
    result = chunker.chunk_hierarchical(md_text)
    
    print(chunker.summary(result))
    
    # Show sample
    if result.parents:
        print("\nSample Parent Chunk:")
        parent = result.parents[0]
        print(f"  ID: {parent.parent_id}")
        print(f"  Tokens: {parent.tokens}")
        print(f"  Children: {len(parent.child_ids)}")
        print(f"  Text: {parent.text[:150]}...")
        
        print("\nSample Child Chunks:")
        for child in result.children[:3]:
            if child.parent_id == parent.parent_id:
                print(f"  - Child {child.child_position}: {child.tokens} tokens")
                print(f"    Text: {child.text[:100]}...")
    
    return result


def test_embedding(chunking_result):
    """Test embedding generation."""
    print("\n" + "="*80)
    print("TEST 3: Embedding Generation")
    print("="*80)
    
    embedder = Embedder()
    
    # Extract child texts
    child_texts = [child.text for child in chunking_result.children]
    
    print(f"\nEmbedding {len(child_texts)} child chunks...")
    embeddings = embedder.embed_batch(child_texts, show_progress=True)
    
    print(f"✅ Embeddings generated")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dimension: {embeddings.shape[1]}")
    print(f"   Dtype: {embeddings.dtype}")
    
    # Test similarity
    if len(embeddings) >= 2:
        sim = embedder.get_similarity(embeddings[0], embeddings[1])
        print(f"   Sample similarity (chunk 0 vs 1): {sim:.4f}")
    
    return embeddings


def test_faiss_indexing(embeddings, chunking_result, doc_path: Path):
    """Test FAISS dual-index."""
    print("\n" + "="*80)
    print("TEST 4: FAISS Indexing")
    print("="*80)
    
    # Create test index
    builder = FAISSBuilder(
        index_name="test_pipeline",
        use_dual=True
    )
    
    # Build metadata
    metadata = []
    for child in chunking_result.children:
        metadata.append({
            'chunk_id': child.chunk_id,
            'parent_id': child.parent_id,
            'text': child.text,
            'tokens': child.tokens,
            'source_file': str(doc_path),
            'source_type': doc_path.suffix[1:],  # 'pdf', 'docx', 'txt'
            'indexed_at': chunking_result.chunked_at,
            'child_position': child.child_position
        })
    
    print(f"\nAdding {len(metadata)} chunks to delta index...")
    builder.add_to_delta(embeddings, metadata)
    
    # Save
    builder.save()
    
    # Get stats
    stats = builder.get_stats()
    print(f"\n✅ Index built successfully")
    print(f"   Main vectors: {stats['main_index']['vectors']}")
    print(f"   Delta vectors: {stats['delta_index']['vectors']}")
    
    return builder


def test_search(builder, embedder):
    """Test search functionality."""
    print("\n" + "="*80)
    print("TEST 5: Search & Retrieval")
    print("="*80)
    
    # Test queries
    test_queries = [
        "What is the main topic of this document?",
        "methodology approach",
        "results findings conclusion"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: '{query}'")
        
        # Embed query
        query_embedding = embedder.embed_single(query)
        
        # Search
        results = builder.search(query_embedding, k=3)
        
        print(f"Found {len(results)} results:")
        for j, result in enumerate(results, 1):
            print(f"  {j}. Score: {result.score:.4f}")
            print(f"     Chunk ID: {result.chunk_id}")
            print(f"     Text: {result.text[:100]}...")


def cleanup_test_index():
    """Remove test index files."""
    print("\n" + "="*80)
    print("CLEANUP")
    print("="*80)
    
    test_files = [
        config.FAISS_DIR / "test_pipeline_main.index",
        config.FAISS_DIR / "test_pipeline_delta.index",
        config.FAISS_DIR / "test_pipeline.metadata",
        config.FAISS_DIR / "test_pipeline_main.stats",
        config.FAISS_DIR / "test_pipeline_delta.stats"
    ]
    
    removed = 0
    for file_path in test_files:
        if file_path.exists():
            file_path.unlink()
            removed += 1
    
    print(f"Removed {removed} test files")


def main():
    parser = argparse.ArgumentParser(description="Test MRA_v3 core utilities pipeline")
    parser.add_argument(
        '--sample-doc',
        type=Path,
        help='Path to sample document (PDF, DOCX, or TXT)'
    )
    parser.add_argument(
        '--keep-index',
        action='store_true',
        help='Keep test index after completion'
    )
    
    args = parser.parse_args()
    
    # Create sample document if not provided
    if args.sample_doc:
        doc_path = args.sample_doc
        if not doc_path.exists():
            print(f"❌ Document not found: {doc_path}")
            sys.exit(1)
    else:
        # Create a temporary test document
        doc_path = Path("test_sample.txt")
        sample_text = """
# Test Document for MRA_v3 Pipeline

## Introduction

This is a test document to verify the MRA_v3 core utilities pipeline. 
It contains multiple sections and sufficient text to generate meaningful 
chunks for testing purposes.

The document processing module should convert this text to Markdown format
with token counting and metadata headers.

## Methodology

Our approach involves several key steps:

1. Document conversion using PyMuPDF4LLM for PDFs
2. Hierarchical chunking with parent-child strategy
3. Embedding generation using Alibaba GTE model
4. FAISS indexing with dual-index architecture

This methodology ensures high-quality semantic search capabilities while
maintaining efficient incremental updates.

## Results

The pipeline successfully processes documents through all stages:

- Conversion: PDF/DOCX/TXT → Markdown
- Chunking: 2000-token parents, 500-token children
- Embedding: 1024-dimensional vectors
- Indexing: HNSW main + Flat delta

## Discussion

The dual-index architecture provides an optimal balance between search
performance and update efficiency. The main HNSW index offers fast
approximate search, while the delta Flat index enables instant additions.

Periodic merging maintains the main index quality without disrupting
real-time document additions.

## Conclusion

This test verifies that all core utilities function correctly and can
be integrated into the full MRA_v3 system. The pipeline demonstrates
robust document processing, accurate chunking, high-quality embeddings,
and efficient FAISS indexing.

Future work will focus on integrating these utilities with the chat
system, web search, and TwistedPair distortion features.
""" * 3  # Repeat to ensure sufficient tokens
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        print(f"Created temporary test document: {doc_path}")
    
    try:
        # Run pipeline tests
        print("\n" + "="*80)
        print("MRA_v3 CORE UTILITIES PIPELINE TEST")
        print("="*80)
        
        # Test 1: Document processing
        md_text = test_document_processing(doc_path)
        
        # Test 2: Chunking
        chunking_result = test_chunking(md_text)
        
        # Test 3: Embedding
        embeddings = test_embedding(chunking_result)
        
        # Test 4: FAISS indexing
        builder = test_faiss_indexing(embeddings, chunking_result, doc_path)
        
        # Test 5: Search
        embedder = Embedder()  # Reuse for search
        test_search(builder, embedder)
        
        # Summary
        print("\n" + "="*80)
        print("PIPELINE TEST SUMMARY")
        print("="*80)
        print("✅ Document processing: PASSED")
        print("✅ Hierarchical chunking: PASSED")
        print("✅ Embedding generation: PASSED")
        print("✅ FAISS indexing: PASSED")
        print("✅ Search & retrieval: PASSED")
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        if not args.keep_index:
            cleanup_test_index()
        
        # Remove temporary test file
        if not args.sample_doc and doc_path.exists():
            doc_path.unlink()
            print(f"Removed temporary test document")


if __name__ == "__main__":
    main()
