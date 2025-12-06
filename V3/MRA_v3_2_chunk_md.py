#!/usr/bin/env python3
"""
MRA_v3_2_chunk_md.py

Step 2: Chunk markdown files into hierarchical parent/child structure.

Features:
- Hierarchical chunking (2000 token parents, 500 token children)
- Links children to parents via IDs
- 100 token overlap between children
- Character offset tracking
- Progress display with chunk statistics

Input:
- MRA_v3/data/markdown/reference_papers/*.md
- MRA_v3/data/markdown/my_papers/*.md

Output:
- MRA_v3/data/chunks/reference_papers_chunks.jsonl
- MRA_v3/data/chunks/my_papers_chunks.jsonl

Usage:
    # Chunk reference papers
    python MRA_v3_2_chunk_md.py --source reference_papers
    
    # Chunk authored papers
    python MRA_v3_2_chunk_md.py --source my_papers
    
    # Chunk sessions
    python MRA_v3_2_chunk_md.py --source sessions

    # Chunk web_cache
    python MRA_v3_2_chunk_md.py --source web_cache

    # Chunk both
    python MRA_v3_2_chunk_md.py --all
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.chunker import Chunker


# Input/Output directories
MARKDOWN_BASE = Path(__file__).parent / "data" / "markdown"
CHUNKS_BASE = Path(__file__).parent / "data" / "chunks"

REFERENCE_MD_DIR = MARKDOWN_BASE / "reference_papers"
MY_PAPERS_MD_DIR = MARKDOWN_BASE / "my_papers"
SESSIONS_MD_DIR = MARKDOWN_BASE / "sessions"
WEB_CACHE_MD_DIR = MARKDOWN_BASE / "web_cache"

# Create output directory
CHUNKS_BASE.mkdir(parents=True, exist_ok=True)


def get_markdown_files(md_dir: Path) -> List[Path]:
    """Get all markdown files from directory"""
    return sorted(md_dir.glob("*.md"))


def chunk_documents(source_name: str, verbose: bool = False):
    """
    Chunk markdown files into hierarchical structure.
    
    Args:
        source_name: 'reference_papers', 'my_papers', 'sessions', or 'web_cache'
        verbose: Show detailed progress
    """
    # Setup paths
    if source_name == "reference_papers":
        md_dir = REFERENCE_MD_DIR
        output_file = CHUNKS_BASE / "reference_papers_chunks.jsonl"
        label = "Reference Papers"
    elif source_name == "my_papers":
        md_dir = MY_PAPERS_MD_DIR
        output_file = CHUNKS_BASE / "my_papers_chunks.jsonl"
        label = "My Authored Papers"
    elif source_name == "sessions":
        md_dir = SESSIONS_MD_DIR
        output_file = CHUNKS_BASE / "sessions_chunks.jsonl"
        label = "Chat Sessions"
    elif source_name == "web_cache":
        md_dir = WEB_CACHE_MD_DIR
        output_file = CHUNKS_BASE / "web_cache_chunks.jsonl"
        label = "Web Cache"
    else:
        raise ValueError(f"Unknown source: {source_name}")
    
    print("="*80)
    print(f"STEP 2: MARKDOWN → CHUNKS ({label})")
    print("="*80)
    print(f"Input:   {md_dir}")
    print(f"Output:  {output_file}")
    print()
    
    # Get markdown files
    md_files = get_markdown_files(md_dir)
    
    if not md_files:
        print(f"⚠️  No markdown files found in {md_dir}")
        print(f"   Run Step 1 first: python MRA_v3_1_pdf_to_md.py --source {source_name}")
        return
    
    print(f"Found {len(md_files)} markdown files")
    if verbose:
        for md_file in md_files:
            print(f"  - {md_file.name}")
    print()
    
    # Initialize chunker
    chunker = Chunker(
        parent_size=2000,
        child_size=500,
        overlap=100
    )
    
    # Track results
    all_chunks = []
    total_parents = 0
    total_children = 0
    failed = []
    
    # Process each markdown file
    for i, md_path in enumerate(md_files, 1):
        print(f"[{i}/{len(md_files)}] 📝 {md_path.name:<60}", end=" ", flush=True)
        
        try:
            # Read markdown
            with open(md_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
            
            # Chunk document
            result = chunker.chunk_hierarchical(md_text, doc_id=md_path.stem)
            
            # Store chunks with metadata (only children are indexed)
            for child in result.children:
                chunk_dict = {
                    'chunk_id': child.chunk_id,
                    'parent_id': child.parent_id,
                    'text': child.text,
                    'tokens': child.tokens,
                    'source_file': md_path.name,
                    'source_type': 'pdf',
                    'doc_id': md_path.stem,
                    'child_position': child.child_position,
                    'start_offset': child.start_offset,
                    'end_offset': child.end_offset
                }
                all_chunks.append(chunk_dict)
            
            # Update counters
            parents = result.total_parents
            children = result.total_children
            total_parents += parents
            total_children += children
            
            print(f"✅ ({parents} parents, {children} children)")
            
        except Exception as e:
            failed.append({'file': md_path.name, 'error': str(e)})
            print(f"❌ {e}")
    
    # Save all chunks to JSONL
    if all_chunks:
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Processed:       {len(md_files) - len(failed)} files")
    if failed:
        print(f"❌ Failed:          {len(failed)} files")
    print(f"\n📊 Total parents:   {total_parents:,}")
    print(f"📊 Total children:  {total_children:,} (indexed in FAISS)")
    print(f"📁 Output file:     {output_file}")
    
    # Show failures
    if failed:
        print(f"\n❌ Failed files:")
        for f in failed:
            print(f"   - {f['file']}: {f['error']}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Chunk markdown into hierarchical parent/child structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chunk reference papers only
  python MRA_v3_2_chunk_md.py --source reference_papers
  
  # Chunk authored papers only
  python MRA_v3_2_chunk_md.py --source my_papers
  
  # Chunk both
  python MRA_v3_2_chunk_md.py --all
        """
    )
    
    parser.add_argument(
        '--source',
        choices=['reference_papers', 'my_papers', 'sessions', 'web_cache'],
        help='Chunk specific source'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Chunk all sources'
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
        chunk_documents(args.source, verbose=args.verbose)
    elif args.all:
        chunk_documents('reference_papers', verbose=args.verbose)
        chunk_documents('my_papers', verbose=args.verbose)
    
    # Show total time
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print()
    print("✅ Step 2 complete! Next step:")
    print("   python MRA_v3_3_embed_chunks.py --all")


if __name__ == "__main__":
    main()
