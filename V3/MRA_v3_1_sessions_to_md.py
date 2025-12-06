#!/usr/bin/env python3
"""
MRA_v3_1_sessions_to_md.py

Step 1: Convert session JSON files to Markdown format.

Converts session logs from data/sessions/*.json to markdown files
in data/markdown/sessions/ for indexing.

Output:
- data/sessions/*.json → data/markdown/sessions/*.md

Usage:
    python MRA_v3_1_sessions_to_md.py

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
SESSIONS_JSON_DIR = Path(__file__).parent / "data" / "sessions"
SESSIONS_MD_DIR = Path(__file__).parent / "data" / "markdown" / "sessions"

# Create output directory
SESSIONS_MD_DIR.mkdir(parents=True, exist_ok=True)


def session_to_markdown(session_file: Path) -> str:
    """
    Convert session JSON to markdown format.
    
    Format matches document structure for chunking:
    - Title and metadata header
    - Conversation turns (User/Assistant)
    - Sources if present
    """
    with open(session_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract data
    title = data.get('title', 'Untitled Session') or 'Untitled Session'
    session_id = data.get('session_id', 'unknown')
    created_at = data.get('created_at', 'unknown')
    updated_at = data.get('updated_at', 'unknown')
    messages = data.get('messages', [])
    
    # Build markdown
    lines = [
        f"# {title}",
        "",
        f"**Session ID:** `{session_id}`  ",
        f"**Created:** {created_at}  ",
        f"**Updated:** {updated_at}  ",
        f"**Messages:** {len(messages)}",
        "",
        "---",
        ""
    ]
    
    # Add conversation
    for i, msg in enumerate(messages, 1):
        role = msg.get('role', 'unknown').capitalize()
        content = msg.get('content', '')
        timestamp = msg.get('timestamp', '')
        
        lines.append(f"## Turn {i}: {role}")
        if timestamp:
            lines.append(f"*{timestamp}*")
        lines.append("")
        lines.append(content)
        lines.append("")
        
        # Add sources if present
        metadata = msg.get('metadata', {})
        sources = metadata.get('sources', [])
        if sources:
            lines.append("**Sources Referenced:**")
            lines.append("")
            for src in sources:
                filename = src.get('filename', 'unknown')
                score = src.get('score', 0)
                doc_id = src.get('doc_id', '')
                lines.append(f"- `{filename}` (score: {score:.3f}, doc: {doc_id})")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def convert_sessions(resume: bool = False, verbose: bool = False):
    """
    Convert all session JSON files to markdown.
    
    Args:
        resume: Skip already converted files
        verbose: Show detailed progress
    """
    print("="*80)
    print("STEP 1: SESSION JSON → MARKDOWN")
    print("="*80)
    print(f"Input:  {SESSIONS_JSON_DIR}")
    print(f"Output: {SESSIONS_MD_DIR}")
    print()
    
    # Get session files
    session_files = sorted(SESSIONS_JSON_DIR.glob("*.json"))
    
    if not session_files:
        print("⚠️  No session files found")
        return
    
    print(f"📁 Found {len(session_files)} session files")
    print()
    
    # Convert
    stats = {'success': 0, 'skipped': 0, 'failed': 0}
    
    for i, session_file in enumerate(session_files, 1):
        # Output filename: use session_id from filename (first part before underscore)
        parts = session_file.stem.split('_')
        session_id = parts[0]
        output_file = SESSIONS_MD_DIR / f"{session_id}.md"
        
        # Check if already converted
        if resume and output_file.exists():
            stats['skipped'] += 1
            if verbose:
                print(f"[{i}/{len(session_files)}] ⏭️  Skipped: {session_file.name}")
            continue
        
        try:
            # Convert to markdown
            md_text = session_to_markdown(session_file)
            
            # Save
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(md_text)
            
            stats['success'] += 1
            if verbose:
                print(f"[{i}/{len(session_files)}] ✅ {session_file.name} → {output_file.name}")
        
        except Exception as e:
            stats['failed'] += 1
            print(f"[{i}/{len(session_files)}] ❌ {session_file.name}: {e}")
    
    # Summary
    print()
    print("="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    print(f"✅ Success:  {stats['success']}")
    print(f"⏭️  Skipped:  {stats['skipped']}")
    print(f"❌ Failed:   {stats['failed']}")
    print(f"📊 Total:    {len(session_files)}")
    print()
    
    if stats['success'] > 0:
        print("✅ Step 1 complete! Next step:")
        print("   python MRA_v3_2_chunk_md.py --source sessions")


def main():
    parser = argparse.ArgumentParser(description="Convert session JSON to markdown")
    
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
    convert_sessions(resume=args.resume, verbose=args.verbose)
    
    # Show total time
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"⏱️  Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
