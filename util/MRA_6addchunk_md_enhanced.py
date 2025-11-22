"""
MRA_6addchunk_md_enhanced.py

The 6th step to prepare MRA data for embedding and indexing.
(The 5th step was to test the MRA operation but is not part of the pipeline.)

Prepares and chunks markdown log files for embedding.
"""

import os
import re
import json
import tiktoken
import unicodedata
import argparse

# === CONFIG ===
INPUT_DIR = "logs"
OUTPUT_FILE = "addchunk_md.jsonl"
LOG_FILE = "log_addchunk_md.txt"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
ENCODER = tiktoken.get_encoding("cl100k_base")

# === HELPERS ===
def tokenize(text):
    return ENCODER.encode(text)

def detokenize(tokens):
    return ENCODER.decode(tokens)

def clean(text, strip_non_ascii=False):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = re.sub(r"\$.*?\$", "", text)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", "", text, flags=re.DOTALL)
    remove_chars = ["\u00ad", "\u201c", "\u201d", "\u2018", "\u2019", "\u2013", "\u2014", "\u2026", "±"]
    for ch in remove_chars:
        text = text.replace(ch, "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Sm" and not unicodedata.name(ch, "").startswith("GREEK"))
    if strip_non_ascii:
        text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def detect_source_type(filename):
    if filename.endswith(".md"):
        return "log" if filename.startswith("MRAlog_") else "pdf"
    return "unknown"

def normalize_headings(text):
    return re.sub(r"(#{1,6})\s*\*{0,2}(.+?)\*{0,2}", r"\1 \2", text)

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

def extract_log_turns(md_text):
    turn_pattern = r"(## Turn \d+)\s+\*\*Timestamp:\*\* (.*?)\s+(?=## Turn|\Z)"
    matches = re.findall(turn_pattern, md_text, flags=re.DOTALL)
    turns = []
    for heading, body in matches:
        section = heading.strip()
        timestamp_match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", body)
        timestamp = timestamp_match.group(0) if timestamp_match else None
        user_match = re.search(r"User:\s*(.*?)\n", body, flags=re.DOTALL)
        assistant_match = re.search(r"\*\*Response:\*\*\s*(.*?)(?=\n\*\*Sources:\*\*|\n\*\*Citations:\*\*|\Z)", body, flags=re.DOTALL)
        user_query = user_match.group(1).strip() if user_match else None
        assistant_response = assistant_match.group(1).strip() if assistant_match else None
        turns.append((section, body.strip(), timestamp, user_query, assistant_response))
    return turns

def score_chunk_quality(text):
    token_count = len(tokenize(text))
    line_count = text.count("\n") + 1
    avg_tokens_per_line = token_count / line_count if line_count else 0
    return round(avg_tokens_per_line, 2)

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

def process_file(path, debug=False, strip_non_ascii=False):
    filename = os.path.basename(path)
    source_type = detect_source_type(filename)
    conversation_id = filename.replace("MRAlog_", "").replace(".md", "") if source_type == "log" else None

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean(raw, strip_non_ascii=strip_non_ascii)
    normalized = normalize_headings(cleaned)

    if source_type == "log":
        sections = extract_log_turns(normalized)
    else:
        sections = extract_sections(normalized)

    all_chunks = []
    for sec_idx, section_data in enumerate(sections):
        if source_type == "log":
            title, content, timestamp, user_query, assistant_response = section_data
        else:
            title, content = section_data
            timestamp = user_query = assistant_response = None

        token_count = len(tokenize(content))
        subchunks = chunk_text_by_tokens_safe(content) if token_count > CHUNK_SIZE else [content]

        for i, chunk in enumerate(subchunks):
            chunk = clean(chunk, strip_non_ascii=strip_non_ascii)
            chunk_data = {
                "filename": filename,
                "source_type": source_type,
                "section": title,
                "chunk_index": f"{sec_idx}_{i}",
                "text": chunk,
                "quality_score": score_chunk_quality(chunk)
            }
            if timestamp:
                chunk_data["timestamp"] = timestamp
            if conversation_id:
                chunk_data["conversation_id"] = conversation_id
            if user_query:
                chunk_data["user_query"] = user_query
            if assistant_response:
                chunk_data["assistant_response"] = assistant_response
            all_chunks.append(chunk_data)

    return all_chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ascii-only", action="store_true")
    args = parser.parse_args()

    all_chunks = []
    for fname in os.listdir(INPUT_DIR):
        if fname.endswith(".md"):
            path = os.path.join(INPUT_DIR, fname)
            chunks = process_file(path, debug=args.debug, strip_non_ascii=args.ascii_only)
            all_chunks.extend(chunks)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk) + "\n")

    print(f"\n✅ Done. {len(all_chunks)} chunks written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()