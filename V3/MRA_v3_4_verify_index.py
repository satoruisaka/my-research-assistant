#!/usr/bin/env python3
"""
MRA_v3_4_verify_index.py

Step 4: Verify FAISS indices by testing retrieval.

Features:
- Tests semantic search against built indices
- Shows top-K results with scores
- Verifies index integrity
- Displays index statistics

Usage:
    # Test reference papers index
    python MRA_v3_4_verify_index.py --source reference_papers --query "reinforcement learning"
    
    # Test authored papers index
    python MRA_v3_4_verify_index.py --source my_papers --query "robot autonomy"
    
    # Test both indices
    python MRA_v3_4_verify_index.py --all --query "neural networks"
    
    # Interactive mode
    python MRA_v3_4_verify_index.py --source reference_papers --interactive

    # Verify sessions index
    python MRA_v3_4_verify_index.py --source sessions --query "chat history"
    
    # Verify web_cache index
    python MRA_v3_4_verify_index.py --source web_cache --query "machine learning"

"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.embedder import Embedder
from utils.faiss_builder import FAISSBuilder


def verify_index(source_name: str, query: str, top_k: int = 5, verbose: bool = False):
    """
    Verify FAISS index by testing retrieval.
    
    Args:
        source_name: 'reference_papers', 'my_papers', 'sessions', or 'web_cache'
        query: Test query string
        top_k: Number of results to show
        verbose: Show detailed information
    """
    # Setup
    if source_name == "reference_papers":
        label = "Reference Papers"
        use_dual = True
    elif source_name == "sessions":
        label = "Chat Sessions"
        use_dual = True
    elif source_name == "web_cache":
        label = "Web Cache"
        use_dual = True
    elif source_name == "my_papers":
        label = "My Authored Papers"
        use_dual = False
    else:
        raise ValueError(f"Unknown source: {source_name}")
    
    print("="*80)
    print(f"STEP 4: VERIFY INDEX ({label})")
    print("="*80)
    print(f"Query: '{query}'")
    print(f"Top-K: {top_k}\n")
    
    try:
        # Load index
        print("📂 Loading index...")
        builder = FAISSBuilder(
            index_name=source_name,
            use_dual=use_dual
        )
        
        # Show statistics
        stats = builder.get_stats()
        print(f"   ✅ Index loaded")
        
        if use_dual:
            print(f"   Main index: {stats['main_index']['vectors']:,} vectors")
            print(f"   Delta index: {stats['delta_index']['vectors']:,} vectors")
        else:
            print(f"   Index: {stats['main_index']['vectors']:,} vectors")
        
        print()
        
        # Initialize embedder
        print("🔧 Loading embedder...")
        embedder = Embedder()
        print(f"   ✅ {embedder.model_name}\n")
        
        # Embed query
        print("🔍 Searching...")
        query_embedding = embedder.embed_single(query)
        
        # Search
        results = builder.search(query_embedding, k=top_k)
        
        print(f"   Found {len(results)} results\n")
        
        # Display results
        print("="*80)
        print("RESULTS")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {result.score:.4f}")
            print(f"    Chunk ID: {result.chunk_id}")
            print(f"    Source: {result.source_file or 'unknown'}")
            print(f"    Doc ID: {result.doc_id or 'unknown'}")
            
            # Show text preview
            text = result.text
            if len(text) > 200:
                text = text[:200] + "..."
            print(f"    Text: {text}")
            
            if verbose:
                print(f"    Parent ID: {result.parent_id}")
                print(f"    Source Type: {result.source_type}")
                print(f"    Section: {result.section_title or 'N/A'}")
        
        print("\n" + "="*80)
        print("✅ Index verification complete!")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"❌ Index not found: {e}")
        print(f"   Run Step 3 first: python MRA_v3_3_embed_chunks.py --source {source_name}")
    except Exception as e:
        print(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def interactive_mode(source_name: str, top_k: int = 5):
    """Interactive query mode"""
    # Setup
    if source_name == "reference_papers":
        label = "Reference Papers"
        use_dual = True
    elif source_name == "my_papers":
        label = "My Authored Papers"
        use_dual = False
    else:
        raise ValueError(f"Unknown source: {source_name}")
    
    print("="*80)
    print(f"INTERACTIVE SEARCH MODE ({label})")
    print("="*80)
    print("Type your queries (or 'quit' to exit)\n")
    
    # Load index and embedder once
    try:
        print("📂 Loading index and embedder...")
        builder = FAISSBuilder(index_name=source_name, use_dual=use_dual)
        embedder = Embedder()
        print("✅ Ready!\n")
        
        while True:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Search
            query_embedding = embedder.embed_single(query)
            results = builder.search(query_embedding, k=top_k)
            
            print(f"\n{len(results)} results:\n")
            
            for i, result in enumerate(results, 1):
                print(f"[{i}] {result.score:.4f} | {result.source_file or 'unknown'}")
                text = result.text
                if len(text) > 150:
                    text = text[:150] + "..."
                print(f"    {text}\n")
            
            print("-"*80 + "\n")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Verify FAISS indices by testing retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test specific index
  python MRA_v3_4_verify_index.py --source reference_papers --query "deep learning"
  
  # Test both indices
  python MRA_v3_4_verify_index.py --all --query "autonomous systems"
  
  # Interactive mode
  python MRA_v3_4_verify_index.py --source reference_papers --interactive
        """
    )
    
    parser.add_argument(
        '--source',
        choices=['reference_papers', 'my_papers', 'sessions', 'web_cache'],
        help='Test specific index'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all indices'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Test query string'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive query mode'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of results to show (default: 5)'
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
    
    if args.interactive and args.all:
        print("⚠️  Interactive mode only works with --source, not --all")
        sys.exit(1)
    
    if args.interactive:
        interactive_mode(args.source, top_k=args.top_k)
    elif args.query:
        if args.source:
            verify_index(args.source, args.query, top_k=args.top_k, verbose=args.verbose)
        elif args.all:
            verify_index('reference_papers', args.query, top_k=args.top_k, verbose=args.verbose)
            print()
            verify_index('my_papers', args.query, top_k=args.top_k, verbose=args.verbose)
            print()
            verify_index('sessions', args.query, top_k=args.top_k, verbose=args.verbose)
            print()
            verify_index('web_cache', args.query, top_k=args.top_k, verbose=args.verbose)
    else:
        parser.print_help()
        print("\n⚠️  Please specify --query or --interactive")


if __name__ == "__main__":
    main()
