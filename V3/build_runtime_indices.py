#!/usr/bin/env python3
"""
build_runtime_indices.py

Build FAISS indices for sessions and web_cache (runtime data).

This script processes session logs and web cache files:
1. Converts JSON session logs to markdown format
2. Chunks the text using hierarchical chunking
3. Embeds chunks and builds FAISS indices

Usage:
    # Build sessions index
    python build_runtime_indices.py --source sessions
    
    # Build web_cache index
    python build_runtime_indices.py --source web_cache
    
    # Build both
    python build_runtime_indices.py --all
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from utils.chunker import Chunker
from utils.embedder import Embedder
from utils.faiss_builder import FAISSBuilder
from config import FAISS_DIR


def session_to_markdown(session_file: Path) -> str:
    """Convert session JSON to markdown format."""
    with open(session_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build markdown
    lines = [
        f"# Session: {data.get('title', 'Untitled Session')}",
        f"",
        f"**Session ID:** {data.get('session_id', 'unknown')}",
        f"**Created:** {data.get('created_at', 'unknown')}",
        f"**Updated:** {data.get('updated_at', 'unknown')}",
        f"",
        "---",
        ""
    ]
    
    # Add conversation
    for msg in data.get('messages', []):
        role = msg.get('role', 'unknown').capitalize()
        content = msg.get('content', '')
        timestamp = msg.get('timestamp', '')
        
        lines.append(f"## {role} ({timestamp})")
        lines.append("")
        lines.append(content)
        lines.append("")
        
        # Add sources if present
        if 'metadata' in msg and 'sources' in msg['metadata']:
            sources = msg['metadata']['sources']
            if sources:
                lines.append("**Sources:**")
                for src in sources:
                    lines.append(f"- {src.get('filename', 'unknown')} (score: {src.get('score', 0):.3f})")
                lines.append("")
    
    return "\n".join(lines)


def web_cache_to_markdown(cache_file: Path) -> str:
    """
    Convert web cache file to markdown.
    
    Assumes cache files are already markdown or JSON with markdown content.
    """
    with open(cache_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to parse as JSON first
    try:
        data = json.loads(content)
        # Extract markdown content if JSON
        if isinstance(data, dict):
            return data.get('content', content)
        return content
    except json.JSONDecodeError:
        # Already markdown
        return content


def chunk_document(md_text: str, doc_id: str, chunker: Chunker) -> List[Dict]:
    """Chunk markdown text into hierarchical chunks."""
    result = chunker.chunk_hierarchical(md_text, doc_id=doc_id)
    
    # Convert to dict format matching MRA_v3_2 output
    chunks = []
    for parent in result.parents:
        for child in parent.children:
            chunks.append({
                'doc_id': doc_id,
                'parent_id': parent.id,
                'child_id': child.id,
                'parent_text': parent.text,
                'child_text': child.text,
                'parent_tokens': parent.token_count,
                'child_tokens': child.token_count,
                'metadata': child.metadata
            })
    
    return chunks


def build_index_for_source(source_name: str, verbose: bool = False):
    """
    Build FAISS index for sessions or web_cache.
    
    Args:
        source_name: 'sessions' or 'web_cache'
        verbose: Show detailed progress
    """
    print("="*80)
    print(f"BUILDING INDEX: {source_name.upper()}")
    print("="*80)
    
    # Setup paths
    data_dir = Path(__file__).parent / "data" / source_name
    if not data_dir.exists():
        print(f"⚠️  Directory not found: {data_dir}")
        return
    
    # Get files
    if source_name == "sessions":
        files = list(data_dir.glob("*.json"))
        converter = session_to_markdown
    elif source_name == "web_cache":
        files = list(data_dir.glob("*.md")) + list(data_dir.glob("*.json"))
        converter = web_cache_to_markdown
    else:
        raise ValueError(f"Unknown source: {source_name}")
    
    if not files:
        print(f"⚠️  No files found in {data_dir}")
        return
    
    print(f"📁 Found {len(files)} files")
    print()
    
    # Initialize processors
    chunker = Chunker()
    embedder = Embedder()
    builder = FAISSBuilder(index_name=source_name, use_dual=True)
    
    # Process files
    all_chunks = []
    all_texts = []
    
    print("📝 Processing files...")
    for i, file_path in enumerate(files, 1):
        if verbose:
            print(f"   [{i}/{len(files)}] {file_path.name}")
        
        try:
            # Convert to markdown
            md_text = converter(file_path)
            
            # Chunk
            chunks = chunk_document(md_text, str(file_path.stem), chunker)
            
            # Collect child chunks for embedding
            for chunk in chunks:
                all_chunks.append(chunk)
                all_texts.append(chunk['child_text'])
        
        except Exception as e:
            print(f"   ⚠️  Failed to process {file_path.name}: {e}")
            continue
    
    if not all_chunks:
        print("❌ No chunks created")
        return
    
    print(f"✅ Created {len(all_chunks):,} chunks")
    print()
    
    # Embed chunks
    print("🔮 Embedding chunks...")
    
    if len(all_texts) > 1000:
        print("   Using streaming mode (large dataset)...")
        embeddings = []
        batch_size = 32
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            batch_embeddings = embedder.embed(batch)
            embeddings.extend(batch_embeddings)
            if verbose or (i // batch_size) % 10 == 0:
                print(f"   Progress: {i+len(batch)}/{len(all_texts)} ({100*(i+len(batch))/len(all_texts):.1f}%)")
    else:
        print("   Using batch mode (small dataset)...")
        embeddings = embedder.embed(all_texts)
    
    embeddings = np.array(embeddings)
    print(f"✅ Generated {len(embeddings):,} embeddings (shape: {embeddings.shape})")
    print()
    
    # Build FAISS index
    print("🔨 Building FAISS index...")
    
    # Add to delta index
    ids = list(range(len(all_chunks)))
    builder.add_to_delta(
        embeddings=embeddings,
        ids=ids,
        metadata_list=all_chunks
    )
    
    print(f"✅ Added {len(all_chunks):,} vectors to delta index")
    
    # Save index
    builder.save()
    print(f"✅ Saved index to {FAISS_DIR}/{source_name}_*.index")
    print()
    
    # Show stats
    stats = builder.get_stats()
    print("📊 Index Statistics:")
    print(f"   Main index:  {stats['main_index']['vectors']:,} vectors")
    print(f"   Delta index: {stats['delta_index']['vectors']:,} vectors")
    print(f"   Total:       {stats['total_vectors']:,} vectors")
    print()


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indices for runtime data")
    
    parser.add_argument(
        '--source',
        choices=['sessions', 'web_cache'],
        help='Build index for specific source'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Build all runtime indices'
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
        build_index_for_source(args.source, verbose=args.verbose)
    elif args.all:
        build_index_for_source('sessions', verbose=args.verbose)
        build_index_for_source('web_cache', verbose=args.verbose)
    
    # Show total time
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print()
    print("✅ Runtime indices built!")
    print("\nVerify with:")
    print("   python MRA_v3_4_verify_index.py --source sessions --interactive")
    print("   python MRA_v3_4_verify_index.py --source web_cache --interactive")


if __name__ == "__main__":
    import numpy as np
    main()
