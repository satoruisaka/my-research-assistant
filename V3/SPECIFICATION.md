# MRA_v3 Specification

**Project:** My Research Assistant v3  
**Version:** 3.0.0  
**Date:** December 2025  
**Status:** In Development

## Executive Summary

MRA_v3 is a local-first RAG (Retrieval-Augmented Generation) research assistant designed for academic paper management, chat-based Q&A, web search integration, and optional rhetorical distortion. All LLM operations run locally via Ollama. The system manages 4 specialized FAISS indices with automatic and manual update strategies.

### Core Capabilities
1. **Document Search** - Unified search across reference papers (823), authored papers (7), sessions, and web cache
2. **Web Search** - BRAVE API (primary) + DuckDuckGo (fallback) with automatic indexing
3. **Chat Sessions** - Multi-turn conversations with Ollama LLMs, persistent logging
4. **Rhetorical Distortion** - Optional integration with TwistedPair V2 API (6 modes × 5 tones × 10 gain levels)
5. **Hybrid RAG** - Combines local FAISS search + web search + distortion in unified workflow

---

## 1. System Architecture

### 1.1 Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI Server                      │
│                     (server.py - port 8000)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │   Web UI     │  │   Chat      │  │   Retrieval       │   │
│  │ (index.html) │→ │   Manager   │→ │   Manager         │   │
│  └──────────────┘  └─────────────┘  └───────────────────┘   │
│                           ↓                    ↓            │
│                    ┌──────────────┐    ┌─────────────────┐  │
│                    │   Ollama     │    │  FAISS Indices  │  │
│                    │   Client     │    │  (4 indices)    │  │
│                    └──────────────┘    └─────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Web Search  │  │  TwistedPair │  │  Auto Indexer     │  │
│  │  (BRAVE+DDG) │  │  Client      │  │  (Background)     │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
           ↓                    ↓                       ↓
    ┌─────────────┐     ┌──────────────┐     ┌────────────────┐
    │   Ollama    │     │  TwistedPair │     │  External Docs │
    │   Server    │     │  V2 Server   │     │  (MyReferences)│
    │ (port 11434)│     │  (port 8001) │     │  (MyPapers)    │
    └─────────────┘     └──────────────┘     └────────────────┘
```

### 1.2 Data Flow

**Document Ingestion:**
```
PDF/DOCX/TXT → Document Processor → Markdown (token-counted)
                                          ↓
                                    Hierarchical Chunker
                                    (2000-token parents)
                                          ↓
                                    Embedder (Alibaba GTE)
                                    (1024-dim vectors)
                                          ↓
                                    FAISS Builder
                                    (dual-index or flat)
                                          ↓
                                    Index + Metadata (pickle)
```

**Search & Retrieval:**
```
User Query → Embedder → FAISS Search (4 indices in parallel)
                                ↓
                        Merge & Rerank Results
                                ↓
                        Retrieve Parent Chunks
                                ↓
                        Context Builder → Ollama LLM → Response
```

**Chat Session:**
```
User Message → Context Builder
                    ↓
              FAISS Search (if RAG enabled)
                    ↓
              Web Search (if enabled)
                    ↓
              TwistedPair Distortion (if enabled)
                    ↓
              Ollama LLM → Assistant Response
                    ↓
              Session Log → Auto-Indexer
```

---

## 2. FAISS Index Architecture

### 2.1 Index Specifications

| Index Name | Type | Size | Update Strategy | Purpose |
|------------|------|------|-----------------|---------|
| **reference_papers** | Dual (HNSW + Flat) | ~1,000 PDFs | Manual (weekly/monthly) | Academic papers from external directory |
| **my_papers** | Single (Flat) | ~12 PDFs | Manual (as needed) | User's authored papers |
| **sessions** | Dual (HNSW + Flat) | Growing | Automatic (on session end) | Chat history for context |
| **web_cache** | Dual (HNSW + Flat) | Growing | Automatic (on web fetch) | Web search results |

### 2.2 Dual-Index Architecture

**Structure:**
```
reference_papers/
├── reference_papers_main.index      # HNSW (IVF4096,Flat)
├── reference_papers_delta.index     # Flat (incremental adds)
├── reference_papers.metadata        # Unified metadata (pickle)
├── reference_papers_main.stats      # Build timestamp, doc count
└── reference_papers_delta.stats     # Incremental stats
```

**Search Algorithm:**
```python
def search_dual_index(index_name, query_embedding, k):
    # 1. Search main index (HNSW)
    main_scores, main_ids = main_index.search(query_embedding, k)
    
    # 2. Search delta index (Flat)
    delta_scores, delta_ids = delta_index.search(query_embedding, k)
    
    # 3. Merge results (adjust IDs for delta offset)
    delta_ids_adjusted = delta_ids + len(main_metadata)
    merged = merge_results(main_scores, main_ids, delta_scores, delta_ids_adjusted)
    
    # 4. Sort by score and return top-k
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged[:k]
```

**Merge Trigger:**
```python
# When delta exceeds 10% of main or 100 documents
if len(delta_metadata) >= max(0.1 * len(main_metadata), 100):
    rebuild_main_index(index_name)  # Merge delta → main
```

### 2.3 Metadata Schema

**Format:** Python pickle (binary) for 2x compression vs JSON

**Structure:**
```python
{
    'metadata': [
        {
            'chunk_id': 'uuid-v4',              # Unique chunk identifier
            'parent_id': 'uuid-v4',             # Parent chunk ID (hierarchical)
            'doc_id': 'uuid-v4',                # Source document ID
            'text': 'chunk content...',         # Full text (for retrieval)
            'source_file': '/abs/path.pdf',     # Original file path
            'file_hash': 'sha256:abc123...',    # SHA256 hash for change detection
            'indexed_at': '2025-12-04T10:30:00Z',
            'source_type': 'pdf',               # pdf|session_log|web_cache|docx|txt
            'section_title': 'Introduction',    # Extracted from Markdown headers
            'tokens': 500,                      # Token count (tiktoken)
            'embedding': [1024 floats],         # Optional: cached for fast rebuild
            'metadata_version': '3.0.0'
        },
        ...
    ],
    'chunk_id_to_idx': {
        'uuid-1': 0,
        'uuid-2': 1,
        ...
    }
}
```

**Size Estimates:**
- Without embeddings: ~60-80MB for 823 papers (~30K chunks)
- With embeddings: ~140-160MB (2x larger, enables fast rebuild)

---

## 3. Chunking Strategy

### 3.1 Hierarchical Chunking

**Parameters:**
- Parent chunk: 2000 tokens (tiktoken `cl100k_base`)
- Child chunk: 500 tokens
- Overlap: 100 tokens

**Algorithm (from TwistedPair V4):**
```python
def chunk_hierarchical(text: str, parent_size: int = 2000, 
                       child_size: int = 500, overlap: int = 100):
    """
    Returns:
        parents: List of parent chunks (2000 tokens each)
        children_per_parent: Dict[parent_id -> List[child_chunks]]
    """
    tokens = tokenizer.encode(text)
    parents = []
    children_per_parent = {}
    
    # Step 1: Split into parents
    for i in range(0, len(tokens), parent_size):
        parent_tokens = tokens[i:i + parent_size]
        parent_id = str(uuid.uuid4())
        parents.append({
            'id': parent_id,
            'text': tokenizer.decode(parent_tokens),
            'tokens': len(parent_tokens),
            'start': i,
            'end': i + len(parent_tokens)
        })
        
        # Step 2: Split parent into children
        children = []
        for j in range(0, len(parent_tokens), child_size - overlap):
            child_tokens = parent_tokens[j:j + child_size]
            if len(child_tokens) < 50:  # Skip tiny chunks
                continue
            children.append({
                'id': str(uuid.uuid4()),
                'parent_id': parent_id,
                'text': tokenizer.decode(child_tokens),
                'tokens': len(child_tokens)
            })
        
        children_per_parent[parent_id] = children
    
    return parents, children_per_parent
```

**Rationale:**
- **Children indexed**: Smaller chunks = better semantic precision
- **Parents returned**: Larger chunks = complete context for LLM
- **Overlap**: Prevents context loss at boundaries

### 3.2 Embedding Strategy

**Only embed children:**
- Reduces FAISS index size by ~75%
- Maintains search precision (children are more focused)
- Parents stored in metadata for retrieval

**Memory efficiency:**
```
Without hierarchical: 10,000 chunks × 1024-dim = 40MB embeddings
With hierarchical: 2,500 parents → 10,000 children
  → Index: 10,000 × 1024-dim = 40MB (same)
  → But: 2,500 parent chunks stored separately (not embedded)
  → Benefit: Better search precision with same memory
```

---

## 4. Embedding Model

### 4.1 Specifications

**Model:** Alibaba-NLP/gte-large-en-v1.5  
**Source:** Hugging Face `sentence-transformers`  
**Dimensions:** 1024  
**Context Window:** 8192 tokens  
**MTEB Score:** 65.4% (as of Nov 2024)  
**License:** MIT  

**Selected for:**
- 1.2% better MTEB performance vs BAAI/bge-large-en-v1.5 (64.2%)
- Alibaba's proven track record with Qwen LLM
- Excellent retrieval performance on academic documents

### 4.2 Batch Processing

**Embedder configuration:**
```python
class Embedder:
    def __init__(self, model_name: str = "Alibaba-NLP/gte-large-en-v1.5"):
        self.model = SentenceTransformer(model_name, device='cuda')
        self.batch_size = 32  # Adjust based on GPU memory
    
    def embed_batch(self, texts: List[str], 
                    show_progress: bool = True) -> np.ndarray:
        """Stream embeddings in batches to manage memory."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        return embeddings  # Shape: (N, 1024)
```

**Performance:**
- GPU (RTX 3090): ~500 chunks/sec
- CPU (12-core): ~50 chunks/sec

---

## 5. API Endpoints

### 5.1 Chat

**POST /api/chat/message**
```json
Request:
{
    "session_id": "uuid-or-new",
    "message": "What is quantum entanglement?",
    "settings": {
        "use_rag": true,
        "use_web_search": false,
        "use_distortion": false,
        "search_scope": {
            "reference_papers": true,
            "my_papers": true,
            "sessions": true,
            "web_cache": false
        },
        "model": "mistral:latest",
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_k_retrieval": 5
    },
    "distortion": {
        "mode": "ECHO_ER",
        "tone": "FORMAL",
        "gain": 5
    }
}

Response:
{
    "session_id": "uuid",
    "message_id": "uuid",
    "assistant_response": "Quantum entanglement is...",
    "context_used": [
        {
            "source": "reference_papers",
            "doc_id": "uuid",
            "snippet": "...",
            "score": 0.92
        }
    ],
    "distorted": false,
    "timestamp": "2025-12-04T10:30:00Z"
}
```

**POST /api/chat/end-session**
```json
Request:
{
    "session_id": "uuid"
}

Response:
{
    "status": "success",
    "session_id": "uuid",
    "messages_count": 15,
    "indexed": true,
    "log_path": "data/sessions/session_uuid.json"
}
```

### 5.2 Search

**POST /api/search**
```json
Request:
{
    "query": "neural network attention mechanisms",
    "scope": {
        "reference_papers": true,
        "my_papers": false,
        "sessions": false,
        "web_cache": false
    },
    "k": 10,
    "filters": {
        "date_range": {
            "start": "2020-01-01",
            "end": "2025-12-31"
        },
        "source_files": ["paper1.pdf", "paper2.pdf"]
    }
}

Response:
{
    "results": [
        {
            "chunk_id": "uuid",
            "parent_id": "uuid",
            "doc_id": "uuid",
            "text": "Attention mechanisms allow...",
            "source_file": "/mnt/c/.../paper.pdf",
            "section_title": "Methodology",
            "score": 0.94,
            "index_name": "reference_papers"
        }
    ],
    "total_found": 10,
    "query_time_ms": 150
}
```

### 5.3 Web Search

**POST /api/web-search**
```json
Request:
{
    "query": "latest transformer architecture 2025",
    "num_results": 10,
    "cache": true
}

Response:
{
    "results": [
        {
            "title": "Transformers v2 Architecture",
            "url": "https://...",
            "snippet": "...",
            "source": "brave",
            "cached": true,
            "indexed": true
        }
    ],
    "total_found": 10,
    "fallback_used": false,
    "query_time_ms": 1200
}
```

### 5.4 Document Management

**POST /api/documents/add**
```json
Request:
{
    "file_path": "/path/to/paper.pdf",
    "index_name": "reference_papers",
    "metadata_override": {
        "author": "Smith et al.",
        "year": 2024
    }
}

Response:
{
    "status": "success",
    "doc_id": "uuid",
    "chunks_added": 45,
    "index_updated": "delta"
}
```

**POST /api/documents/rebuild-index**
```json
Request:
{
    "index_name": "reference_papers"
}

Response:
{
    "status": "success",
    "index_name": "reference_papers",
    "rebuild_time_sec": 120.5,
    "total_chunks": 30000,
    "delta_merged": true
}
```

### 5.5 Distortion (TwistedPair Integration)

**POST /api/distort**
```json
Request:
{
    "text": "Original content to distort",
    "mode": "INVERT_ER",
    "tone": "CASUAL",
    "gain": 7,
    "model": "mistral:latest"
}

Response:
{
    "original": "Original content...",
    "distorted": "Distorted version...",
    "mode": "INVERT_ER",
    "tone": "CASUAL",
    "gain": 7,
    "model_used": "mistral:latest",
    "timestamp": "2025-12-04T10:30:00Z"
}
```

---

## 6. Update Mechanisms

### 6.1 Manual Updates (Papers)

**Script:** `update_paper_indices.py`

**Usage:**
```bash
# Check for new/changed files (dry run)
python update_paper_indices.py --check

# Update reference papers only
python update_paper_indices.py --index reference_papers

# Update authored papers only
python update_paper_indices.py --index my_papers

# Update all paper indices
python update_paper_indices.py --all

# Verbose mode
python update_paper_indices.py --all --verbose
```

**Algorithm:**
1. Scan source directory for PDFs
2. Compute SHA256 hashes
3. Compare with existing metadata
4. Process new/changed files → delta index
5. Log changes

**Change detection:**
```python
def detect_changes(source_dir: Path, existing_metadata: dict):
    existing_hashes = {m['source_file']: m['file_hash'] 
                       for m in existing_metadata}
    
    new_files = []
    changed_files = []
    
    for pdf_path in source_dir.glob('**/*.pdf'):
        current_hash = compute_sha256(pdf_path)
        
        if str(pdf_path) not in existing_hashes:
            new_files.append(pdf_path)
        elif existing_hashes[str(pdf_path)] != current_hash:
            changed_files.append(pdf_path)
    
    return new_files, changed_files
```

### 6.2 Automatic Updates (Sessions & Web Cache)

**Trigger:** Background task via `auto_indexer.py` (singleton)

**Session indexing:**
```python
# In chat_manager.py
def end_session(session_id: str):
    log_path = save_session_log(session_id)
    
    if config.AUTO_INDEX_SESSIONS:
        auto_indexer.index_session_log(
            log_path=log_path,
            session_id=session_id
        )
```

**Web cache indexing:**
```python
# In web_search.py
def fetch_and_cache(url: str, query: str):
    content = fetch_url(url)
    cache_path = save_to_cache(url, content, query)
    
    if config.AUTO_INDEX_WEB_CACHE:
        auto_indexer.index_web_cache(
            cache_path=cache_path,
            url=url,
            query=query
        )
```

**Auto-indexer implementation:**
```python
class AutoIndexer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def index_session_log(self, log_path: Path, session_id: str):
        """Process session log in background thread."""
        thread = threading.Thread(
            target=self._process_session,
            args=(log_path, session_id),
            daemon=True
        )
        thread.start()
    
    def _process_session(self, log_path: Path, session_id: str):
        try:
            # 1. Load session log
            with open(log_path) as f:
                log_data = json.load(f)
            
            # 2. Extract text (user + assistant messages)
            text = self._extract_session_text(log_data)
            
            # 3. Chunk
            parents, children = chunker.chunk_hierarchical(text)
            
            # 4. Embed
            embeddings = embedder.embed_batch([c['text'] for c in children])
            
            # 5. Add to delta index
            faiss_builder.add_to_delta(
                embeddings=embeddings,
                metadata=self._build_metadata(children, log_path, session_id),
                index_name='sessions'
            )
            
            logger.info(f"Indexed session {session_id}: {len(children)} chunks")
        except Exception as e:
            logger.error(f"Auto-indexing failed: {e}")
```

---

## 7. Configuration

### 7.1 Environment Variables

**File:** `.env` (git-ignored)

```bash
# Required
BRAVE_API_KEY=your_brave_api_key_here

# Optional: Directory overrides
MRA_REFERENCE_PAPERS_DIR=/mnt/c/Users/sator/Documents/MyReferences
MRA_MY_PAPERS_DIR=/mnt/c/Users/sator/Documents/MyAuthoredPapers

# Optional: Service URLs
OLLAMA_URL=http://localhost:11434
TWISTEDPAIR_URL=http://localhost:8000

# Optional: Feature flags
MRA_AUTO_INDEX_SESSIONS=true
MRA_AUTO_INDEX_WEB_CACHE=true
MRA_AUTO_INDEX_PAPERS=false
```

### 7.2 config.py Constants

```python
# Paths (external sources)
SOURCE_REFERENCE_DIR = Path(os.getenv(
    'MRA_REFERENCE_PAPERS_DIR',
    '/mnt/c/Users/sator/Documents/MyReferences'
))
SOURCE_AUTHORED_DIR = Path(os.getenv(
    'MRA_MY_PAPERS_DIR',
    '/mnt/c/Users/sator/Documents/MyAuthoredPapers'
))

# Paths (internal)
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
FAISS_DIR = PROJECT_ROOT / 'faiss_indices'
SESSIONS_DIR = DATA_DIR / 'sessions'
WEB_CACHE_DIR = DATA_DIR / 'web_cache'

# Embedding
EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 32

# Chunking
PARENT_CHUNK_SIZE = 2000
CHILD_CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOKENIZER = "cl100k_base"  # tiktoken

# FAISS
FAISS_NLIST = 4096  # IVF clusters for HNSW
FAISS_NPROBE = 64   # Search clusters
DELTA_MERGE_THRESHOLD = 100  # Max docs in delta before merge
DELTA_MERGE_RATIO = 0.1      # Merge when delta > 10% of main

# Services
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
TWISTEDPAIR_URL = os.getenv('TWISTEDPAIR_URL', 'http://localhost:8000')
BRAVE_API_KEY = os.getenv('BRAVE_API_KEY', '')

# Auto-indexing
AUTO_INDEX_SESSIONS = os.getenv('MRA_AUTO_INDEX_SESSIONS', 'true').lower() == 'true'
AUTO_INDEX_WEB_CACHE = os.getenv('MRA_AUTO_INDEX_WEB_CACHE', 'true').lower() == 'true'
AUTO_INDEX_PAPERS = os.getenv('MRA_AUTO_INDEX_PAPERS', 'false').lower() == 'true'

# API
API_HOST = '0.0.0.0'
API_PORT = 8001
API_CORS_ORIGINS = ['http://localhost:8001', 'http://127.0.0.1:8001']

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = PROJECT_ROOT / 'mra_v3.log'
```

---

## 8. Error Handling

### 8.1 Custom Exceptions (errors.py)

**Exception Hierarchy:**
```
MRAError (base)
├── FileNotFoundError
├── DocumentProcessingError
├── ChunkingError
├── EmbeddingError
├── FAISSIndexError
├── NetworkError
│   ├── OllamaConnectionError
│   ├── TwistedPairConnectionError
│   └── WebSearchError
├── ConfigurationError
└── DependencyError
```

**Usage:**
```python
from errors import (
    DocumentProcessingError, handle_error, 
    retry_on_failure, safe_operation
)

@safe_operation(error_category=ErrorCategory.PROCESSING)
def process_document(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Processing logic...

@retry_on_failure(max_retries=3, delay=2.0, exceptions=(NetworkError,))
def call_ollama(prompt: str):
    response = requests.post(OLLAMA_URL, json={'prompt': prompt})
    response.raise_for_status()
    return response.json()
```

### 8.2 Error Response Format (API)

```json
{
    "error": {
        "code": "DOCUMENT_PROCESSING_ERROR",
        "message": "Failed to convert PDF to Markdown",
        "category": "PROCESSING",
        "context": {
            "file_path": "/path/to/paper.pdf",
            "operation": "pdf_to_markdown"
        },
        "timestamp": "2025-12-04T10:30:00Z",
        "request_id": "uuid"
    }
}
```

---

## 9. Performance Specifications

### 9.1 Throughput Targets

| Operation | Target | Measured (RTX 3090) |
|-----------|--------|---------------------|
| PDF → Markdown | 1-5 sec/page | - |
| Chunking (10K tokens) | <100ms | - |
| Embedding (batch=32) | <2 sec | - |
| FAISS search (HNSW) | <200ms | - |
| FAISS search (Flat) | <50ms | - |
| Ollama generation (2K tokens) | 5-15 sec | - |
| Full RAG pipeline | 2-5 sec | - |

### 9.2 Memory Requirements

| Component | RAM | VRAM (GPU) |
|-----------|-----|------------|
| Base application | 500MB | - |
| Embedding model | 1.5GB | 2GB |
| FAISS indices (all) | 2GB | - |
| Ollama (mistral 7B) | 4GB | 8GB |
| **Total (GPU mode)** | **8GB** | **10GB** |
| **Total (CPU mode)** | **12GB** | **0GB** |

### 9.3 Disk Requirements

| Data | Size |
|------|------|
| Reference papers (823 PDFs) | ~2.5GB |
| Authored papers (7 PDFs) | ~50MB |
| FAISS indices (all) | ~500MB |
| Metadata (pickle) | ~80MB |
| Embedding model cache | ~1.3GB |
| Session logs (1000 sessions) | ~100MB |
| Web cache (1000 pages) | ~200MB |
| **Total** | **~4.7GB** |

---

## 10. Security & Privacy

### 10.1 Data Privacy

**Local-First Design:**
- All LLM processing via local Ollama (no cloud APIs)
- All embeddings computed locally
- Documents never leave user's machine
- Web search via BRAVE (privacy-focused) or DuckDuckGo

**API Key Security:**
- `.env` file git-ignored
- BRAVE_API_KEY never logged
- No telemetry or analytics

### 10.2 Input Validation

**API requests:**
```python
from pydantic import BaseModel, Field, validator

class ChatRequest(BaseModel):
    session_id: str = Field(..., regex=r'^[0-9a-f-]{36}$')
    message: str = Field(..., min_length=1, max_length=10000)
    
    @validator('message')
    def sanitize_message(cls, v):
        # Strip potential injection patterns
        return v.strip()
```

### 10.3 File System Safety

**Prevent path traversal:**
```python
def validate_file_path(path: Path, allowed_dirs: List[Path]):
    resolved = path.resolve()
    if not any(resolved.is_relative_to(d) for d in allowed_dirs):
        raise SecurityError(f"Path outside allowed directories: {path}")
    return resolved
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

**Modules to test:**
- `document_processor.py` - PDF/DOCX/TXT conversion
- `chunker.py` - Hierarchical chunking logic
- `embedder.py` - Embedding generation
- `faiss_builder.py` - Index operations
- `retrieval_manager.py` - Search & merge logic

**Example:**
```python
# test_chunker.py
def test_hierarchical_chunking():
    text = "..." * 10000  # Long text
    parents, children = chunker.chunk_hierarchical(text)
    
    assert len(parents) > 0
    assert all(p['tokens'] <= 2000 for p in parents)
    assert all(len(children[p['id']]) > 0 for p in parents)
    assert all(c['tokens'] <= 500 for child_list in children.values() 
               for c in child_list)
```

### 11.2 Integration Tests

**Full pipeline test:**
```python
# test_pipeline.py
def test_document_to_faiss():
    # 1. Add document
    doc_id = add_document('test.pdf', 'reference_papers')
    
    # 2. Search for content
    results = search('neural networks', scope={'reference_papers': True})
    
    # 3. Verify retrieval
    assert any(r['doc_id'] == doc_id for r in results)
```

### 11.3 Performance Tests

**Benchmarking:**
```python
# test_performance.py
def test_search_latency():
    query = "quantum computing applications"
    
    start = time.time()
    results = retrieval_manager.unified_search(query, k=10)
    latency = (time.time() - start) * 1000
    
    assert latency < 500  # <500ms target
    assert len(results) <= 10
```

---

## 12. Deployment

### 12.1 System Requirements

**Minimum:**
- OS: Ubuntu 20.04+ (via WSL on Windows)
- CPU: 8 cores (for CPU-only mode)
- RAM: 16GB
- Disk: 10GB free
- Python: 3.10+

**Recommended:**
- OS: Ubuntu 22.04 (WSL 2)
- CPU: 12+ cores
- RAM: 32GB
- GPU: NVIDIA RTX 3090 or better (24GB VRAM)
- Disk: 20GB free
- Python: 3.11

### 12.2 Installation

```bash
# 1. Clone repository
cd /mnt/c/Users/sator/Documents/linuxproject
git clone <repo> MRA_v3
cd MRA_v3

# 2. Create virtual environment
python3.11 -m venv .venv
source ./.venv/bin/activate

# 3. Install numpy first (critical)
pip install numpy==1.26.0

# 4. Install system dependencies (Ubuntu)
sudo apt update
sudo apt install -y \
    tesseract-ocr pngquant ghostscript \
    cmake g++ libjpeg-dev libpng-dev libtiff-dev zlib1g-dev

# 5. Install Python dependencies
pip install -r requirements.txt

# 6. Configure environment
cp .env.example .env
nano .env  # Add BRAVE_API_KEY

# 7. Initialize indices (first time)
python update_paper_indices.py --all

# 8. Start Ollama (separate terminal)
ollama serve

# 9. Start MRA_v3
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### 12.3 Ollama Setup

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull models
ollama pull mistral:latest
ollama pull llama3.1:latest
ollama pull gemma2:9b

# Verify
ollama list
```

### 12.4 TwistedPair V2 Setup (Optional)

```bash
cd /mnt/c/Users/sator/Documents/linuxproject/TwistedPair/V2
source ./.venv/bin/activate
uvicorn server:app --port 8000 --reload
```

---

## 13. Maintenance

### 13.1 Regular Tasks

| Task | Frequency | Command |
|------|-----------|---------|
| Check new papers | Weekly | `python update_paper_indices.py --check` |
| Update paper indices | Weekly | `python update_paper_indices.py --all` |
| Rebuild main indices | Monthly | `python utils/rebuild_indices.py --all` |
| Clean old sessions | Monthly | `python utils/cleanup_sessions.py --older-than 90d` |
| Backup indices | Weekly | `tar -czf backup.tar.gz faiss_indices/` |

### 13.2 Monitoring

**Log files:**
- `mra_v3.log` - Application logs
- `data/sessions/` - Chat session logs
- `faiss_indices/*.stats` - Index statistics

**Health check endpoint:**
```bash
curl http://localhost:8001/api/health
```

**Response:**
```json
{
    "status": "healthy",
    "services": {
        "ollama": "connected",
        "twistedpair": "connected",
        "faiss_indices": {
            "reference_papers": {"status": "ok", "chunks": 30000},
            "my_papers": {"status": "ok", "chunks": 150},
            "sessions": {"status": "ok", "chunks": 5000},
            "web_cache": {"status": "ok", "chunks": 2000}
        }
    },
    "uptime_seconds": 3600,
    "version": "3.0.0"
}
```

---

## 14. Future Enhancements

### 14.1 Planned Features (v3.1)

1. **Multi-modal search** - Images from papers (via CLIP)
2. **Graph RAG** - Entity extraction + knowledge graph
3. **Citation tracking** - Link cited papers automatically
4. **Export to BibTeX** - Generate citations from retrieved papers
5. **Collaborative sessions** - Multi-user chat rooms

### 14.2 Research Directions

1. **Hybrid search** - Combine dense (FAISS) + sparse (BM25) retrieval
2. **Reranking** - Cross-encoder for result refinement
3. **Active learning** - User feedback → improve retrieval
4. **Query expansion** - LLM-generated query variations

---

## 15. References

### 15.1 Related Projects

- **TwistedPair V2** - Rhetorical distortion API
- **TwistedPair V4** - Hierarchical chunking & dual-index pattern
- **MyResearchAssistant v2** - Previous iteration (deprecated)

### 15.2 External Dependencies

- [Ollama](https://ollama.ai) - Local LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [sentence-transformers](https://www.sbert.net/) - Embedding models
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing

### 15.3 Papers & Resources

- **RAG:** [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- **FAISS:** [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
- **GTE:** [General Text Embeddings](https://arxiv.org/abs/2308.03281)

---

## Appendix A: File Structure

```
MRA_v3/
├── .env                         # Environment variables (git-ignored)
├── .env.example                 # Template
├── .gitignore                   # Git ignore rules
├── config.py                    # Configuration
├── errors.py                    # Error handling
├── requirements.txt             # Python dependencies
├── README.md                    # Setup guide
├── SPECIFICATION.md             # This document
├── INDEX_UPDATE_GUIDE.md        # Index management guide
├── GOTCHAS_RESOLVED.md          # Known issues tracker
├── server.py                    # FastAPI main server
├── chat_manager.py              # Chat session management
├── retrieval_manager.py         # Unified FAISS search
├── web_search.py                # BRAVE + DuckDuckGo
├── twistedpair_client.py        # TwistedPair V2 client
├── ollama_client.py             # Ollama LLM interface
├── auto_indexer.py              # Auto-indexing (sessions/web)
├── update_paper_indices.py      # Manual paper indexing
├── .github/
│   └── copilot-instructions.md  # AI agent instructions
├── static/
│   ├── index.html               # Web UI
│   ├── styles.css
│   └── app.js
├── data/
│   ├── sessions/                # Chat logs (auto-indexed)
│   │   ├── session_uuid1.json
│   │   └── ...
│   └── web_cache/               # Web search results (auto-indexed)
│       ├── cache_uuid1.json
│       └── ...
├── faiss_indices/
│   ├── reference_papers_main.index
│   ├── reference_papers_delta.index
│   ├── reference_papers.metadata
│   ├── reference_papers_main.stats
│   ├── reference_papers_delta.stats
│   ├── my_papers.index
│   ├── my_papers.metadata
│   ├── my_papers.stats
│   ├── sessions_main.index
│   ├── sessions_delta.index
│   ├── sessions.metadata
│   ├── sessions_main.stats
│   ├── sessions_delta.stats
│   ├── web_cache_main.index
│   ├── web_cache_delta.index
│   ├── web_cache.metadata
│   ├── web_cache_main.stats
│   └── web_cache_delta.stats
├── utils/
│   ├── document_processor.py    # PDF/DOCX/TXT → Markdown
│   ├── chunker.py               # Hierarchical chunking
│   ├── embedder.py              # Alibaba GTE embeddings
│   ├── faiss_builder.py         # Dual-index manager
│   ├── rebuild_indices.py       # Manual rebuild tool
│   ├── cleanup_sessions.py      # Cleanup old sessions
│   └── test_pipeline.py         # Integration tests
└── models/                      # Embedding model cache (auto-downloaded)
    └── Alibaba-NLP/
        └── gte-large-en-v1.5/
```

---

## Appendix B: Glossary

- **RAG** - Retrieval-Augmented Generation (search + LLM)
- **FAISS** - Facebook AI Similarity Search (vector index)
- **HNSW** - Hierarchical Navigable Small World (FAISS index type)
- **IVF** - Inverted File Index (FAISS clustering)
- **Flat** - Brute-force exact search (FAISS index type)
- **Embedding** - Vector representation of text (1024-dim for GTE)
- **Chunk** - Text segment (parent: 2000 tokens, child: 500 tokens)
- **Metadata** - Chunk information stored alongside embeddings
- **Delta index** - Temporary index for new documents before merge
- **Main index** - Primary HNSW index (optimized for search)
- **Ollama** - Local LLM inference server
- **TwistedPair** - Rhetorical distortion system
- **BRAVE** - Privacy-focused web search API
- **DDG** - DuckDuckGo search engine (fallback)
- **MTEB** - Massive Text Embedding Benchmark (quality metric)
- **GTE** - General Text Embeddings (Alibaba model)

---

**Document Version:** 1.0.0  
**Last Updated:** December 4, 2025  
**Status:** Draft (in development)
