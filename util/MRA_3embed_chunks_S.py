"""
MRA_3embed_chunks_S.py (stream version to handle large files)

The 3rd step in the MRA data preparation pipeline.

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
import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Config
CHUNK_FILE = "chunk_md.jsonl"
INDEX_PATH = "faiss_index"
METADATA_PATH = "chunk_metadata.json"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32

def load_model(name):
    try:
        print("🔧 Loading embedding model...")
        return SentenceTransformer(name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{name}': {e}")

def stream_chunks(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunk file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def embed_chunks(model, chunk_iter, batch_size):
    embeddings = []
    metadata = []
    batch = []
    for chunk in tqdm(chunk_iter, desc="🧠 Embedding chunks"):
        text = chunk["text"].strip().replace("\n", " ")
        batch.append(text)
        metadata.append(chunk)
        if len(batch) == batch_size:
            emb = model.encode(batch)
            embeddings.append(emb)
            batch = []
    if batch:
        emb = model.encode(batch)
        embeddings.append(emb)
    return np.vstack(embeddings), metadata

def build_index(vectors):
    dim = vectors.shape[1]
    print("📊 Building FAISS index...")
    index = faiss.IndexHNSWFlat(dim, 32)  # Scalable alternative to IndexFlatL2
    index.add(vectors)
    return index

def save_index(index, path):
    faiss.write_index(index, path)
    print(f"✅ FAISS index saved to {path}")

def save_metadata(metadata, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to {path}")

def main():
    try:
        model = load_model(MODEL_NAME)
        print("loaded model:", MODEL_NAME)

        chunk_iter = stream_chunks(CHUNK_FILE)
        print("loaded chunk file:")

        vectors, metadata = embed_chunks(model, chunk_iter, BATCH_SIZE)
        print("embedded all chunks.")

        index = build_index(vectors)
        print("built FAISS index.")
        save_index(index, INDEX_PATH)
        print("saved FAISS index.")
        save_metadata(metadata, METADATA_PATH)
        print("saved metadata.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()