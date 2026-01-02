"""
config.py

MRA_v3 Configuration - Unified configuration with environment variable support
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === PROJECT PATHS ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_DIR = PROJECT_ROOT / "faiss_indices"
MODELS_DIR = PROJECT_ROOT / "models"

# Create internal directories
for dir_path in [DATA_DIR, FAISS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Internal data directories (created automatically)
WEB_CACHE_DIR = DATA_DIR / "web_cache"
SESSIONS_DIR = DATA_DIR / "sessions"

# Create internal directories
for dir_path in [WEB_CACHE_DIR, SESSIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)



# === OLLAMA SETTINGS ===
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_API_GENERATE = f"{OLLAMA_URL}/api/generate"
OLLAMA_API_CHAT = f"{OLLAMA_URL}/api/chat"

# Available LLM models (must be installed: ollama list)
# AVAILABLE_LLM_MODELS = ["qwen3:latest"]

# === LLM model SETTINGS ===
# MAX_CHAT_HISTORY_TURNS = 10  # Keep last N turns in context
DEFAULT_MODEL = "ministral-3:latest"
NUM_CTX = 128000  # Context window (tokens) - keep this value not too high for low GPU machines

# Default output tokens
DEFAULT_OUTPUT_TOKENS = 8000
MAX_OUTPUT_TOKENS = 32000

# Default sampling parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPEAT_PENALTY = 1.1

# Maximum allowed values (for validation)
MAX_TEMPERATURE = 5.0
MAX_TOP_P = 0.98
MAX_TOP_K = 120

FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0

# === TWISTEDPAIR V2 INTEGRATION ===
TWISTEDPAIR_URL = os.getenv("TWISTEDPAIR_URL", "http://localhost:8001")
TWISTEDPAIR_DISTORT_ENDPOINT = f"{TWISTEDPAIR_URL}/distort-manual"

# === WEB SEARCH ===
# BRAVE API Key (from .env - REQUIRED)
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
if not BRAVE_API_KEY:
    print("⚠️  WARNING: BRAVE_API_KEY not set in .env file")
    print("   Web search will fail. Copy .env.example to .env and add your key.")

MAX_WEB_RESULTS = 7
WEB_FETCH_TIMEOUT = 10  # seconds
WEB_FETCH_MAX_CHARS = 10000  # per URL

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"
# User-Agent rotation to avoid bot detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0"
]

# === SOURCE DIRECTORIES ===
# External directories containing source documents
# Override via environment variables if needed

# Reference papers directory (~823 PDFs)
SOURCE_REFERENCE_DIR = Path(os.getenv(
    "REFERENCE_PAPERS_DIR",
    "/home/sator/project/MyReferences"
))

# Your authored papers directory (7 PDFs)
SOURCE_AUTHORED_DIR = Path(os.getenv(
    "MY_PAPERS_DIR",
    "/home/sator/project/MyAuthoredPapers"
))

# Aliases for backward compatibility with update_paper_indices.py
REFERENCE_PAPERS_DIR = SOURCE_REFERENCE_DIR
MY_PAPERS_DIR = SOURCE_AUTHORED_DIR

# Debug: Show what was loaded (commented out to reduce console noise)
# print(f"📁 Reference Papers directory: {SOURCE_REFERENCE_DIR}")
# print(f"📁 Authored Papers directory: {SOURCE_AUTHORED_DIR}")

# Validate external source directories exist
if not SOURCE_REFERENCE_DIR.exists():
    print(f"⚠️  WARNING: Reference Papers directory not found: {SOURCE_REFERENCE_DIR}")
    print(f"   To fix: Create directory or set MRA_REFERENCE_PAPERS_DIR environment variable")

if not SOURCE_AUTHORED_DIR.exists():
    print(f"⚠️  WARNING: Authored Papers directory not found: {SOURCE_AUTHORED_DIR}")
    print(f"   To fix: Create directory or set MRA_MY_PAPERS_DIR environment variable")

# NOTE: New reference papers added via API go to reference_papers_delta.index
# Delta index handles incremental additions without separate directory

# === FAISS INDICES CONFIGURATION ===
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # Alternative: more compatible than Alibaba-NLP
EMBEDDING_DIM = 1024  # bge-large-en-v1.5 dimension
EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation
EMBEDDING_DEVICE = "cuda"  # or "cpu"

# Delta index merge thresholds (for dual-index mode)
DELTA_MERGE_THRESHOLD = 5000  # Merge when delta has this many vectors
DELTA_MERGE_RATIO = 0.1       # Or when delta > 10% of main index size

# Index configurations
INDICES = {
    "reference_papers": {
        "type": "dual",  # main (HNSW) + delta (Flat)
        "source_dir": SOURCE_REFERENCE_DIR,  # 823 PDFs from MyReferences
        "index_path": FAISS_DIR / "reference_papers_main.index",
        "delta_path": FAISS_DIR / "reference_papers_delta.index",
        "metadata_path": FAISS_DIR / "reference_papers.metadata",  # pickle
        "hnsw_m": 32,
        "hnsw_ef_construction": 200,
        "hnsw_ef_search": 100,
        "merge_threshold": 100  # Rebuild when delta > 100 docs
    },
    "my_papers": {
        "type": "single",  # Flat only (small)
        "source_dir": SOURCE_AUTHORED_DIR,  # 7 PDFs from MyAuthoredPapers
        "index_path": FAISS_DIR / "my_papers.index",
        "metadata_path": FAISS_DIR / "my_papers.metadata",  # pickle
    },
    "sessions": {
        "type": "dual",  # main (HNSW) + delta (Flat)
        "source_dir": SESSIONS_DIR,
        "index_path": FAISS_DIR / "sessions_main.index",
        "delta_path": FAISS_DIR / "sessions_delta.index",
        "metadata_path": FAISS_DIR / "sessions.metadata",  # pickle
        "hnsw_m": 32,
        "merge_threshold": 50
    },
    "web_cache": {
        "type": "dual",  # main (HNSW) + delta (Flat)
        "source_dir": WEB_CACHE_DIR,
        "index_path": FAISS_DIR / "web_cache_main.index",
        "delta_path": FAISS_DIR / "web_cache_delta.index",
        "metadata_path": FAISS_DIR / "web_cache.metadata",  # pickle
        "hnsw_m": 32,
        "merge_threshold": 50
    }
}

# === CHUNKING PARAMETERS ===
# Hierarchical chunking strategy (from TwistedPair V4)
PARENT_CHUNK_SIZE = 2000  # tokens
CHILD_CHUNK_SIZE = 500    # tokens
CHUNK_OVERLAP = 100       # tokens
TOKENIZER = "cl100k_base"  # tiktoken model for GPT-4

# === RETRIEVAL SETTINGS ===
TOP_K_RETRIEVAL = 20  # Initial FAISS search
TOP_K_RERANK = 5      # After reranking (optional)
SIMILARITY_THRESHOLD = 0.7  # Minimum score to include

# === CACHE SETTINGS ===
CACHE_EMBEDDINGS = True  # Store embeddings in metadata for fast rebuild
WEB_CACHE_EXPIRE_DAYS = 90  # Prune web cache older than N days

# === FILE MONITORING ===
# Track indexed files by source path and hash
TRACK_SOURCE_FILES = True  # Store source_file and file_hash in metadata

# Auto-indexing behavior (during MRA_v3 runtime)
AUTO_INDEX_SESSIONS = True   # Auto-index when session ends
AUTO_INDEX_WEB_CACHE = True  # Auto-index when web result saved
AUTO_INDEX_PAPERS = False    # Manual: Run update_paper_indices.py

# === LOGGING ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "mra_v3.log"
