#!/usr/bin/env python3
"""
MRA_3embed_chunks_B.py (batch processing for smaller files)

Embeds text chunks using a transformer model and builds a FAISS index for semantic search.

Dependencies:
  pip install -U sentence-transformers
  pip install -U faiss-gpu : for cpu only, use faiss-cpu
  pip install -U tqdm

For GPU setup:
  run nvidia-smi to verify GPU availability and CUDA version
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  where cu121 corresponds to CUDA 12.1
  Then install FAISS with GPU support:
  pip install -U faiss-gpu

For CPU-only setup (optional):
  pip uninstall torch (if installed)
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -U faiss-cpu

Inputs:
  - chunk_md.jsonl : JSONL file with chunk metadata and content

Outputs:
  - faiss_index         : Serialized FAISS index for vector search
  - chunk_metadata.json : Metadata file aligned with index entries

Usage:
  python MRA_3embed_chunks.py
"""
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Config
CHUNK_FILE = "chunk_md.jsonl"
INDEX_PATH = "faiss_index"
METADATA_PATH = "chunk_metadata.json"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Load model
print("🔧 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# Load chunks
print("📦 Loading chunks...")
chunks = []
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

# Normalize and embed
print("🧠 Embedding chunks...")
texts = [chunk["content"].strip().replace("\n", " ") for chunk in chunks]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# Build FAISS index
print("📊 Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Save index
faiss.write_index(index, INDEX_PATH)
print(f"✅ FAISS index saved to {INDEX_PATH}")

# Save metadata
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)
print(f"✅ Metadata saved to {METADATA_PATH}")

