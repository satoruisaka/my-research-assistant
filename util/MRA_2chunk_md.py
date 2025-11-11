"""
MRA_2chunk_md.py

Chunking and cleaning pipeline for Markdown files intended for semantic embedding.

This script processes Markdown documents from the 'pdfs_md' directory, splits them into token-safe chunks,
and applies aggressive symbolic noise removal to optimize for downstream embedding tasks.

Key features:
- Uses OpenAI's cl100k_base tokenizer for token-aware chunking.
- Splits documents by Markdown headings and paragraphs.
- Applies token-length constraints with optional overlap.
- Cleans each chunk by:
  - Normalizing Unicode (NFKC)
  - Removing LaTeX-style math expressions
  - Stripping typographic artifacts (quotes, dashes, ellipses, soft hyphen, ±)
  - Removing all Unicode math symbols and Greek letters
  - Optionally removing all non-ASCII characters (--ascii-only)
- Logs special tokens and exceptions when --debug is enabled.
- Outputs cleaned chunks to 'chunk_md.jsonl' for embedding.

Prep:
    pip install -U tiktoken
Usage:
    python MRA_chunk_md.py
    python MRA_chunk_md.py --debug
    python MRA_chunk_md.py --ascii-only
    python MRA_chunk_md.py --debug --ascii-only
"""
from fileinput import filename
import os
import re
import json
import tiktoken
import unicodedata
import argparse

# === CONFIG ===
INPUT_DIR = "pdfs_md"
OUTPUT_FILE = "chunk_md.jsonl"
LOG_FILE = "log_chunk_md.txt"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
ENCODER = tiktoken.get_encoding("cl100k_base")

# === SPECIAL TOKEN SANITIZATION ===
SPECIAL_TOKEN_PATTERN = r"<\|.*?\|>"

def detect_source_type(filename):
    """
    Infers source type based on filename prefix.
    - Files starting with 'MRAlog_' are treated as logs.
    - All other .md files are assumed to be converted from PDFs.
    """
    if filename.endswith(".md"):
        if filename.startswith("MRAlog_"):
            return "log"
        else:
            return "pdf"
    return "unknown"

def sanitize_special_tokens(text):
    return re.sub(SPECIAL_TOKEN_PATTERN, "", text)

def extract_special_tokens(text):
    return re.findall(SPECIAL_TOKEN_PATTERN, text)

def log_special_tokens(filename, tokens, debug, log_path=LOG_FILE):
    if debug:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{filename}: Special tokens — {', '.join(tokens)}\n")

def log_exception(filename, error, debug, log_path=LOG_FILE):
    if debug:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{filename}: Exception — {type(error).__name__}: {str(error)}\n")

def log_status(filename, section_count, debug, log_path=LOG_FILE):
    if debug:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"Processing {filename} — {section_count} sections\n")
    print(f"Processing: {filename}")

# === HELPERS ===
def tokenize(text):
    return ENCODER.encode(text)

def detokenize(tokens):
    return ENCODER.decode(tokens)

def clean(text, strip_non_ascii=False):
    original = text
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters (except \n and \t)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)

    # Remove LaTeX-style math
    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", "", text, flags=re.DOTALL)

    # Remove typographic artifacts
    remove_chars = ["\u00ad", "\u201c", "\u201d", "\u2018", "\u2019", "\u2013", "\u2014", "\u2026", "±"]
    for ch in remove_chars:
        text = text.replace(ch, "")

    # Remove math symbols and Greek letters
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Sm" and not unicodedata.name(ch, "").startswith("GREEK")
    )

    # Optional: remove all non-ASCII characters
    if strip_non_ascii:
        text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Normalize spacing
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()

def normalize_headings(text):
    return re.sub(r"(#{1,6})\s*\*{0,2}(.+?)\*{0,2}", r"\1 \2", text)

# === CHUNKING ===
def split_by_paragraphs(text, max_tokens):
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    buffer = ""
    for para in paragraphs:
        candidate = buffer + "\n\n" + para if buffer else para
        if len(tokenize(candidate)) <= max_tokens:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = para
    if buffer:
        chunks.append(buffer.strip())
    return chunks

def chunk_text_by_tokens_safe(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = tokenize(text)
    total_tokens = len(tokens)
    chunks = []
    step = chunk_size - overlap
    for start in range(0, total_tokens, step):
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = detokenize(chunk_tokens)
        if len(tokenize(chunk_text)) > chunk_size:
            chunk_text = detokenize(chunk_tokens[:chunk_size])
        chunks.append(chunk_text.strip())
    return chunks

# === SECTION SPLITTING ===
def extract_sections(md_text):
    pattern = r"(#{1,6} ?(?:\*{0,2})?.+?(?:\*{0,2})?)"
    parts = re.split(pattern, md_text)
    sections = []
    if parts and parts[0].strip():
        sections.append(("Untitled", parts[0].strip()))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            sections.append((heading, content))
    return sections

# === MAIN PROCESSING ===
def process_file(path, debug, strip_non_ascii=False):
    filename = os.path.basename(path)
    source_type = detect_source_type(filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        tokens_found = extract_special_tokens(raw)
        if tokens_found:
            log_special_tokens(filename, tokens_found, debug)

        raw = sanitize_special_tokens(raw)
        cleaned = clean(raw, strip_non_ascii=strip_non_ascii)
        normalized = normalize_headings(cleaned)
        sections = extract_sections(normalized)

        if not sections:
            if debug:
                print(f"⚠️ No sections found in {filename}.")
            return []

        log_status(filename, len(sections), debug)

        all_chunks = []
        for sec_idx, (title, content) in enumerate(sections):
            content = clean(content, strip_non_ascii=strip_non_ascii)
            token_count = len(tokenize(content))
            if debug:
                print(f"📚 Section '{title}' — {token_count} tokens")

            if token_count > CHUNK_SIZE:
                prechunks = split_by_paragraphs(content, CHUNK_SIZE)
                subchunks = []
                for pc in prechunks:
                    subchunks.extend(chunk_text_by_tokens_safe(pc, chunk_size=CHUNK_SIZE, overlap=0))
                if debug:
                    print(f"⚠️ Section '{title}' split into {len(subchunks)} chunks")
            else:
                subchunks = [content]
                if debug:
                    print(f"✅ Section '{title}' fits in one chunk")

            for i, chunk in enumerate(subchunks):
                chunk = clean(chunk, strip_non_ascii=strip_non_ascii)
                chunk_tokens = tokenize(chunk)
                if debug:
                    print(f"   🔹 Chunk {sec_idx}_{i}: {len(chunk_tokens)} tokens")
                    if len(chunk_tokens) > CHUNK_SIZE:
                        print(f"   ❌ Oversized chunk detected")
                all_chunks.append({
                    "filename": filename,
                    "source_type": source_type,
                    "section": title,
                    "chunk_index": f"{sec_idx}_{i}",
                    "text": chunk
                })
        return all_chunks

    except Exception as e:
        log_exception(filename, e, debug)
        print(f"❌ Error processing {filename}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Chunk Markdown files for embedding.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    parser.add_argument("--ascii-only", action="store_true", help="Remove all non-ASCII characters")
    args = parser.parse_args()
    debug = args.debug
    strip_non_ascii = args.ascii_only

    if debug:
        open(LOG_FILE, "w").close()

    all_chunks = []
    for fname in os.listdir(INPUT_DIR):
        if fname.endswith(".md"):
            path = os.path.join(INPUT_DIR, fname)
            chunks = process_file(path, debug, strip_non_ascii=strip_non_ascii)
            all_chunks.extend(chunks)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk) + "\n")

    print(f"\n✅ Done. {len(all_chunks)} chunks written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()