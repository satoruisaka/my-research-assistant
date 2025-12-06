#!/usr/bin/env python3
"""
MRA_v3_3_embed_chunks.py

Step 3: Embed chunks and build FAISS indices.

Features:
- Embeds child chunks using BAAI/bge-large-en-v1.5 (1024-dim)
- Builds dual FAISS indices (HNSW main + Flat delta)
- GPU-accelerated with batch processing
- Progress bars and statistics

Input:
- MRA_v3/data/chunks/reference_papers_chunks.jsonl
- MRA_v3/data/chunks/my_papers_chunks.jsonl

Output:
- MRA_v3/faiss_indices/reference_papers_main.index
- MRA_v3/faiss_indices/reference_papers_delta.index
- MRA_v3/faiss_indices/reference_papers.metadata
- MRA_v3/faiss_indices/my_papers.index
- MRA_v3/faiss_indices/my_papers.metadata

Usage:
    # Embed reference papers
    python MRA_v3_3_embed_chunks.py --source reference_papers
    
    # Embed authored papers
    python MRA_v3_3_embed_chunks.py --source my_papers
    
    # Embed sessions
    python MRA_v3_3_embed_chunks.py --source sessions
    
    # Embed web_cache
    python MRA_v3_3_embed_chunks.py --source web_cache

    # Embed both
    python MRA_v3_3_embed_chunks.py --all
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.embedder import Embedder
from utils.faiss_builder import FAISSBuilder
from config import FAISS_DIR


# Input directory
CHUNKS_BASE = Path(__file__).parent / "data" / "chunks"


def load_chunks(chunks_file: Path) -> List[Dict]:
    """Load chunks from JSONL file"""
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_and_index(source_name: str, verbose: bool = False):
    """
    Embed chunks and build FAISS index.
    
    Args:
        source_name: 'reference_papers' or 'my_papers'
        verbose: Show detailed progress
    """
    # Setup paths
    if source_name == "reference_papers":
        chunks_file = CHUNKS_BASE / "reference_papers_chunks.jsonl"
        label = "Reference Papers"
        use_dual = True  # Large corpus uses dual-index
    elif source_name == "my_papers":
        chunks_file = CHUNKS_BASE / "my_papers_chunks.jsonl"
        label = "My Authored Papers"
        use_dual = False  # Small corpus uses single Flat index
    elif source_name == "sessions":
        chunks_file = CHUNKS_BASE / "sessions_chunks.jsonl"
        label = "Chat Sessions"
        use_dual = True  # Dual index for growing dataset
    elif source_name == "web_cache":
        chunks_file = CHUNKS_BASE / "web_cache_chunks.jsonl"
        label = "Web Cache"
        use_dual = True  # Dual index for growing dataset
    else:
        raise ValueError(f"Unknown source: {source_name}")
    
    print("="*80)
    print(f"STEP 3: CHUNKS → EMBEDDINGS → FAISS ({label})")
    print("="*80)
    print(f"Input:   {chunks_file}")
    print(f"Output:  {FAISS_DIR}/")
    print(f"Index:   {'Dual (HNSW + Flat)' if use_dual else 'Single (Flat)'}")
    print()
    
    # Check input file exists
    if not chunks_file.exists():
        print(f"⚠️  Chunks file not found: {chunks_file}")
        print(f"   Run Step 2 first: python MRA_v3_2_chunk_md.py --source {source_name}")
        return
    
    # Load chunks
    print("📦 Loading chunks...")
    chunks = load_chunks(chunks_file)
    print(f"   Loaded {len(chunks):,} chunks\n")
    
    if not chunks:
        print("⚠️  No chunks found in file")
        return
    
    # Initialize embedder
    print("🔧 Initializing embedder...")
    embedder = Embedder()
    print(f"   Model: {embedder.model_name}")
    print(f"   Device: {embedder.device}")
    print(f"   Batch size: {embedder.batch_size}\n")
    
    # Choose embedding method based on dataset size
    chunk_count = len(chunks)
    use_streaming = chunk_count > 1000  # Stream for large datasets
    
    if use_streaming:
        print(f"🧠 Generating embeddings (streaming mode for {chunk_count:,} chunks)...")
        # Stream processing (memory-efficient)
        def text_iterator():
            for chunk in chunks:
                yield chunk['text']
        
        embedding_batches = []
        for batch in embedder.embed_stream(text_iterator(), show_progress=True, normalize=True):
            embedding_batches.append(batch)
        
        embeddings = np.vstack(embedding_batches)
        print(f"   Generated {embeddings.shape[0]:,} embeddings")
        print(f"   Dimension: {embeddings.shape[1]}\n")
    else:
        print(f"🧠 Generating embeddings (batch mode for {chunk_count:,} chunks)...")
        # Batch processing (faster for small datasets)
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedder.embed_batch(
            texts,
            show_progress=True,
            normalize=True  # L2 normalization for cosine similarity
        )
        print(f"   Generated {embeddings.shape[0]:,} embeddings")
        print(f"   Dimension: {embeddings.shape[1]}\n")
    
    # Build FAISS index (load existing if present)
    print("📊 Loading/creating FAISS index...")
    builder = FAISSBuilder(
        index_name=source_name,
        use_dual=use_dual
    )
    
    # Try to load existing index
    index_exists = False
    try:
        builder.load()
        index_exists = True
        print(f"   ✅ Loaded existing index")
        print(f"      Main: {builder.main_index.ntotal} vectors")
        if use_dual:
            print(f"      Delta: {builder.delta_index.ntotal} vectors")
    except (FileNotFoundError, Exception):
        print(f"   ℹ️  No existing index found, creating new one")
    
    # Filter out already-indexed chunks (incremental mode)
    if index_exists:
        new_chunks = []
        new_embeddings = []
        
        for i, chunk in enumerate(chunks):
            if chunk['chunk_id'] not in builder.chunk_id_to_idx:
                new_chunks.append(chunk)
                new_embeddings.append(embeddings[i])
        
        if not new_chunks:
            print(f"\n✅ No new chunks to index (all {len(chunks)} chunks already indexed)")
            return
        
        print(f"\n🔄 Incremental update:")
        print(f"   Total chunks: {len(chunks):,}")
        print(f"   Already indexed: {len(chunks) - len(new_chunks):,}")
        print(f"   New chunks to add: {len(new_chunks):,}\n")
        
        # Use only new chunks/embeddings
        chunks = new_chunks
        embeddings = np.array(new_embeddings)
    
    # Add embeddings (use appropriate method based on index type and creation mode)
    if use_dual:
        if index_exists:
            # Incremental: add to delta
            builder.add_to_delta(embeddings, chunks)
        else:
            # First-time creation: add to main
            print("🏗️  First-time index creation: adding to main HNSW index...")
            builder.add_to_main(embeddings, chunks)
    else:
        # Single-index mode: always add to main
        builder.add_to_main(embeddings, chunks)
    
    # Save
    builder.save()
    
    print(f"   ✅ Index saved\n")
    
    # Show statistics
    stats = builder.get_stats()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Indexed chunks:  {len(chunks):,}")
    print(f"📊 Embedding dim:   {embeddings.shape[1]}")
    
    if use_dual:
        print(f"\n📁 Main index:      {stats['main_index']['vectors']:,} vectors")
        print(f"📁 Delta index:     {stats['delta_index']['vectors']:,} vectors")
    else:
        print(f"\n📁 Index:           {stats['main_index']['vectors']:,} vectors")
    
    print(f"\n💾 Index files:     {FAISS_DIR}/")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Embed chunks and build FAISS indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed reference papers only
  python MRA_v3_3_embed_chunks.py --source reference_papers
  
  # Embed authored papers only
  python MRA_v3_3_embed_chunks.py --source my_papers
  
  # Embed both
  python MRA_v3_3_embed_chunks.py --all
        """
    )
    
    parser.add_argument(
        '--source',
        choices=['reference_papers', 'my_papers', 'sessions', 'web_cache'],
        help='Embed specific source'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Embed all sources'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Validation
    if not any([args.source, args.all]):
        parser.print_help()
        print("\n⚠️  Please specify --source or --all")
        sys.exit(1)
    
    # Execute
    start_time = datetime.now()
    
    if args.source:
        embed_and_index(args.source, verbose=args.verbose)
    elif args.all:
        embed_and_index('reference_papers', verbose=args.verbose)
        embed_and_index('my_papers', verbose=args.verbose)
    
    # Show total time
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print()
    print("✅ Step 3 complete! Next step:")
    print("   python MRA_v3_4_verify_index.py --all")


if __name__ == "__main__":
    main()
