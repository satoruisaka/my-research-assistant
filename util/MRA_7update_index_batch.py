"""
MRA_7update_index_batch.py

The 7th and final step to prepare MRA data for embedding and indexing.

Embeds new chunks and updates existing FAISS index and metadata.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os

# Config
CHUNK_FILE = "addchunk_md.jsonl"
INDEX_PATH = "faiss_index"
METADATA_PATH = "chunk_metadata_updated.json"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Load model
print("🔧 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# Load new chunks
print("📦 Loading new chunks...")
chunks = []
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

# Embed
print(f"🧠 Embedding {len(chunks)} chunks...")
texts = [chunk["text"].strip().replace("\n", " ") for chunk in chunks]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# Load or create FAISS index
print("📊 Updating FAISS index...")
dim = embeddings.shape[1]
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))
faiss.write_index(index, INDEX_PATH)
print(f"✅ FAISS index updated and saved to {INDEX_PATH}")

# Load existing metadata
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        existing_metadata = json.load(f)
else:
    existing_metadata = []

# Append new metadata
new_metadata = []
for chunk in chunks:
    metadata_entry = {
        "filename": chunk["filename"],
        "source_type": chunk.get("source_type", "unknown"),
        "section": chunk.get("section", ""),
        "chunk_index": chunk.get("chunk_index", ""),
        "text": chunk["text"],
        "timestamp": chunk.get("timestamp", datetime.now().isoformat()),
        "quality_score": chunk.get("quality_score", None),
        "conversation_id": chunk.get("conversation_id", None),
        "user_query": chunk.get("user_query", None),
        "assistant_response": chunk.get("assistant_response", None)
    }
    new_metadata.append(metadata_entry)

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(existing_metadata + new_metadata, f, indent=2)
print(f"✅ Metadata updated with {len(new_metadata)} new entries.")