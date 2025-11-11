# retrieval.py
"""
Unified retrieval module: supports FAISS semantic search, metadata filtering, and reranking.
"""

import json
import faiss
import torch
from collections import defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Load models ===
print("🔧 Loading embedder and reranker...")
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")

# === Load index and metadata ===
# INDEX_PATH = "faiss_index"
# METADATA_PATH = "chunk_metadata.json"

def load_index_and_metadata(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# === Metadata utilities ===
def filter_logs(metadata):
    return [m for m in metadata if m.get("source_type") == "log"]

def filter_by_date(metadata, date_str):
    return [m for m in metadata if m.get("timestamp", "").startswith(date_str)]

def group_by_conversation(metadata):
    grouped = defaultdict(list)
    for m in metadata:
        cid = m.get("conversation_id")
        if cid:
            grouped[cid].append(m)
    return grouped

def get_high_quality_chunks(metadata, threshold=40.0):
    return [m for m in metadata if m.get("quality_score", 0) >= threshold]

def format_turn(chunk):
    return f"[{chunk.get('section', '')} — {chunk.get('timestamp', '')}]\nUser: {chunk.get('user_query', '')}\nAssistant: {chunk.get('assistant_response', '')}"

# === FAISS search ===
def search(query, index, metadata, top_k=20):
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        results.append({
            "rank": i + 1,
            "score": float(distances[0][i]),
            "filename": chunk["filename"],
            "source_type": chunk["source_type"],
            "section": chunk["section"],
            "text": chunk["text"],
            "timestamp": chunk.get("timestamp", ""),
            "conversation_id": chunk.get("conversation_id", ""),
            "user_query": chunk.get("user_query", ""),
            "assistant_response": chunk.get("assistant_response", ""),
            "quality_score": chunk.get("quality_score", None)
        })
    return results

# === Reranking ===
def rerank(query, chunks, top_k=5):
    scored = []
    for chunk in chunks:
        inputs = reranker_tokenizer(query, chunk["text"], return_tensors="pt", truncation=True)
        with torch.no_grad():
            score = reranker_model(**inputs).logits.item()  # Single relevance score
        scored.append((score, chunk))
    reranked = sorted(scored, key=lambda x: x[0], reverse=True)
    return [item[1] for item in reranked[:top_k]]

# === Alternative Reranking Approach (sigmoid) ===
#            logits = reranker_model(**inputs).logits
#            score = torch.sigmoid(logits)[0].item()

# === Example usage ===
if __name__ == "__main__":
    index, metadata = load_index_and_metadata()

    query = "define rbot"
    print(f"\n🔍 Searching for: {query}")
    results = search(query, index, metadata, top_k=20)
    reranked = rerank(query, results, top_k=5)

    for chunk in reranked:
        print(format_turn(chunk))
        print("-" * 60)