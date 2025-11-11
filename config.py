# config.py
"""
Centralized configuration for paths, model settings, and token limits.
"""

# Paths
INDEX_PATH = "faiss_index"
METADATA_PATH = "chunk_metadata_updated.json"

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

# Token management
MAX_TOKENS = 3000  # Adjust based on model context window

# Brave web search API key
BRAVE_API_KEY = "your API key"

