# main.py
"""
Entry point for My Research Assistant - the conversational RAG engine using local FAISS + Mistral via Ollama.
"""

from conversation import start_conversation
from retrieval import load_index_and_metadata
from config import INDEX_PATH, METADATA_PATH

def main():
    print("🧠 Initializing Conversational RAG Engine...")
    index, metadata = load_index_and_metadata(INDEX_PATH, METADATA_PATH)
    start_conversation(index, metadata)

if __name__ == "__main__":
    main()