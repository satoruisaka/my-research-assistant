# migrate_chunk_metadata.py

import json
from datetime import datetime

OLD_PATH = "chunk_metadata.json"
NEW_PATH = "chunk_metadata_updated.json"

def infer_source_type(filename):
    ext = filename.split(".")[-1].lower()
    if ext == "md":
        return "pdf"  # assuming all .md files were converted from PDFs
    elif ext == "txt":
        return "txt"
    elif ext == "html":
        return "web"
    elif ext == "csv":
        return "csv"
    elif ext in ["doc", "docx"]:
        return "doc"
    else:
        return "unknown"

def migrate():
    with open(OLD_PATH, "r", encoding="utf-8") as f:
        old_data = json.load(f)

    new_data = []
    for entry in old_data:
        filename = entry.get("source", "unknown.md")
        new_entry = {
            "filename": filename,
            "source_type": infer_source_type(filename),
            "section": entry.get("section", ""),
            "chunk_index": entry.get("chunk_index", ""),
            "text": entry.get("content", ""),
            "timestamp": datetime.now().isoformat()
        }
        new_data.append(new_entry)

    with open(NEW_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)

    print(f"✅ Migrated {len(new_data)} entries to updated format.")

if __name__ == "__main__":
    migrate()