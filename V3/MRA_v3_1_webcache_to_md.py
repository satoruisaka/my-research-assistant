#!/usr/bin/env python3
"""
MRA_v3_1_webcache_to_md.py

Step 1: Convert web cache JSON files to Markdown format.

Converts web search cache from data/web_cache/*.json to markdown files
in data/markdown/web_cache/ for indexing.

Output:
- data/web_cache/*.json → data/markdown/web_cache/*.md

Usage:
    python MRA_v3_1_webcache_to_md.py
    # First run - converts all files
python MRA_v3_1_sessions_to_md.py -v

# Later run - skips already converted files
python MRA_v3_1_sessions_to_md.py --resume -v

# Same for web cache
python MRA_v3_1_webcache_to_md.py --resume -v

"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Input and output directories
WEB_CACHE_JSON_DIR = Path(__file__).parent / "data" / "web_cache"
WEB_CACHE_MD_DIR = Path(__file__).parent / "data" / "markdown" / "web_cache"

# Create output directory
WEB_CACHE_MD_DIR.mkdir(parents=True, exist_ok=True)


def webcache_to_markdown(cache_file: Path) -> str:
    """
    Convert web cache JSON to markdown format.
    
    Format:
    - Search query and metadata header
    - Each search result as a section
    - Title, URL, snippet, source
    """
    with open(cache_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract data
    query = data.get('query', 'Unknown query')
    timestamp = data.get('timestamp', 'unknown')
    results = data.get('results', [])
    
    # Build markdown
    lines = [
        f"# Web Search: {query}",
        "",
        f"**Query:** {query}  ",
        f"**Timestamp:** {timestamp}  ",
        f"**Results:** {len(results)}",
        "",
        "---",
        ""
    ]
    
    # Add search results
    for i, result in enumerate(results, 1):
        title = result.get('title', 'Untitled')
        url = result.get('url', '')
        snippet = result.get('snippet', '')
        source = result.get('source', 'unknown')
        result_timestamp = result.get('timestamp', '')
        
        lines.append(f"## Result {i}: {title}")
        lines.append("")
        if url:
            lines.append(f"**URL:** <{url}>")
        if source:
            lines.append(f"**Source:** {source}")
        if result_timestamp:
            lines.append(f"**Retrieved:** {result_timestamp}")
        lines.append("")
        
        if snippet:
            # Clean up snippet (remove HTML entities)
            snippet_clean = snippet.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            lines.append(snippet_clean)
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def convert_webcache(resume: bool = False, verbose: bool = False):
    """
    Convert all web cache JSON files to markdown.
    
    Args:
        resume: Skip already converted files
        verbose: Show detailed progress
    """
    print("="*80)
    print("STEP 1: WEB CACHE JSON → MARKDOWN")
    print("="*80)
    print(f"Input:  {WEB_CACHE_JSON_DIR}")
    print(f"Output: {WEB_CACHE_MD_DIR}")
    print()
    
    # Get cache files
    cache_files = sorted(WEB_CACHE_JSON_DIR.glob("*.json"))
    
    if not cache_files:
        print("⚠️  No web cache files found")
        return
    
    print(f"📁 Found {len(cache_files)} web cache files")
    print()
    
    # Convert
    stats = {'success': 0, 'skipped': 0, 'failed': 0}
    
    for i, cache_file in enumerate(cache_files, 1):
        # Output filename: same stem as input
        output_file = WEB_CACHE_MD_DIR / f"{cache_file.stem}.md"
        
        # Check if already converted
        if resume and output_file.exists():
            stats['skipped'] += 1
            if verbose:
                print(f"[{i}/{len(cache_files)}] ⏭️  Skipped: {cache_file.name}")
            continue
        
        try:
            # Convert to markdown
            md_text = webcache_to_markdown(cache_file)
            
            # Save
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_text)
            
            stats['success'] += 1
            if verbose:
                print(f"[{i}/{len(cache_files)}] ✅ {cache_file.name} → {output_file.name}")
        
        except Exception as e:
            stats['failed'] += 1
            print(f"[{i}/{len(cache_files)}] ❌ {cache_file.name}: {e}")
    
    # Summary
    print()
    print("="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    print(f"✅ Success:  {stats['success']}")
    print(f"⏭️  Skipped:  {stats['skipped']}")
    print(f"❌ Failed:   {stats['failed']}")
    print(f"📊 Total:    {len(cache_files)}")
    print()
    
    if stats['success'] > 0:
        print("✅ Step 1 complete! Next step:")
        print("   python MRA_v3_2_chunk_md.py --source web_cache")


def main():
    parser = argparse.ArgumentParser(description="Convert web cache JSON to markdown")
    
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
    
    # Execute
    start_time = datetime.now()
    convert_webcache(resume=args.resume, verbose=args.verbose)
    
    # Show total time
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"⏱️  Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
