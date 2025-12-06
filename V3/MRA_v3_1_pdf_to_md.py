#!/usr/bin/env python3
"""
MRA_v3_1_pdf_to_md.py

Step 1: Convert PDFs to Markdown with token counting and metadata.

Features:
- Converts PDFs from MyReferences and MyAuthoredPapers
- Adds token count header
- Includes SHA256 hash and metadata
- OCR for scanned PDFs (via ocrmypdf)
- Progress display with success/failure tracking

Output:
- MyReferences → MRA_v3/data/markdown/reference_papers/
- MyAuthoredPapers → MRA_v3/data/markdown/my_papers/

Usage:
    # Convert reference papers
    python MRA_v3_1_pdf_to_md.py --source reference_papers
    
    # Convert authored papers
    python MRA_v3_1_pdf_to_md.py --source my_papers
    
    # Convert both
    python MRA_v3_1_pdf_to_md.py --all
    
    # Resume (skip already converted)
    python MRA_v3_1_pdf_to_md.py --all --resume
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.document_processor import DocumentProcessor
from config import SOURCE_REFERENCE_DIR, SOURCE_AUTHORED_DIR


# Output directories
MARKDOWN_BASE = Path(__file__).parent / "data" / "markdown"
REFERENCE_MD_DIR = MARKDOWN_BASE / "reference_papers"
MY_PAPERS_MD_DIR = MARKDOWN_BASE / "my_papers"

# Create output directories
REFERENCE_MD_DIR.mkdir(parents=True, exist_ok=True)
MY_PAPERS_MD_DIR.mkdir(parents=True, exist_ok=True)


def get_pdf_files(source_dir: Path):
    """Get all PDF files from directory"""
    return sorted(source_dir.glob("*.pdf"))


def convert_pdfs(source_name: str, resume: bool = False, verbose: bool = False):
    """
    Convert PDFs from source directory to markdown.
    
    Args:
        source_name: 'reference_papers' or 'my_papers'
        resume: Skip already converted files
        verbose: Show detailed progress
    """
    # Setup paths
    if source_name == "reference_papers":
        source_dir = SOURCE_REFERENCE_DIR
        output_dir = REFERENCE_MD_DIR
        label = "Reference Papers"
    elif source_name == "my_papers":
        source_dir = SOURCE_AUTHORED_DIR
        output_dir = MY_PAPERS_MD_DIR
        label = "My Authored Papers"
    else:
        raise ValueError(f"Unknown source: {source_name}")
    
    print("="*80)
    print(f"STEP 1: PDF → MARKDOWN ({label})")
    print("="*80)
    print(f"Source:  {source_dir}")
    print(f"Output:  {output_dir}")
    print()
    
    # Get PDF files
    pdf_files = get_pdf_files(source_dir)
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {source_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files\n")
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Track results
    converted = []
    skipped = []
    failed = []
    
    # Process each PDF
    for i, pdf_path in enumerate(pdf_files, 1):
        md_path = output_dir / f"{pdf_path.stem}.md"
        
        # Check if already exists (resume mode)
        if resume and md_path.exists():
            skipped.append(pdf_path.name)
            if verbose:
                print(f"[{i}/{len(pdf_files)}] ⏭️  SKIP: {pdf_path.name} (already exists)")
            continue
        
        # Show progress
        print(f"[{i}/{len(pdf_files)}] 📄 {pdf_path.name:<60}", end=" ", flush=True)
        
        try:
            # Convert to markdown
            md_text = processor.convert_to_markdown(
                pdf_path,
                include_token_count=True,
                include_metadata=True
            )
            
            # Save markdown
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_text)
            
            # Count tokens for display
            token_count = processor.count_tokens(md_text)
            
            converted.append({
                'file': pdf_path.name,
                'tokens': token_count,
                'output': md_path
            })
            
            print(f"✅ ({token_count:,} tokens)")
            
        except Exception as e:
            failed.append({
                'file': pdf_path.name,
                'error': str(e)
            })
            print(f"❌ {e}")
    
    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Converted:  {len(converted)} files")
    if skipped:
        print(f"⏭️  Skipped:    {len(skipped)} files (already exist)")
    if failed:
        print(f"❌ Failed:     {len(failed)} files")
    
    # Show total tokens
    if converted:
        total_tokens = sum(c['tokens'] for c in converted)
        print(f"\n📊 Total tokens: {total_tokens:,}")
        print(f"📁 Output directory: {output_dir}")
    
    # Show failures
    if failed:
        print(f"\n❌ Failed files:")
        for f in failed:
            print(f"   - {f['file']}: {f['error']}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Convert PDFs to Markdown with token counting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert reference papers only
  python MRA_v3_1_pdf_to_md.py --source reference_papers
  
  # Convert authored papers only
  python MRA_v3_1_pdf_to_md.py --source my_papers
  
  # Convert both
  python MRA_v3_1_pdf_to_md.py --all
  
  # Resume (skip already converted)
  python MRA_v3_1_pdf_to_md.py --all --resume
        """
    )
    
    parser.add_argument(
        '--source',
        choices=['reference_papers', 'my_papers'],
        help='Convert specific source'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Convert all sources'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already converted files'
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
        convert_pdfs(args.source, resume=args.resume, verbose=args.verbose)
    elif args.all:
        convert_pdfs('reference_papers', resume=args.resume, verbose=args.verbose)
        convert_pdfs('my_papers', resume=args.resume, verbose=args.verbose)
    
    # Show total time
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print()
    print("✅ Step 1 complete! Next step:")
    print("   python MRA_v3_2_chunk_md.py --all")


if __name__ == "__main__":
    main()
