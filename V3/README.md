# MRA - My Research Assistant v3

***Personal soundboard and information retrieval with local LLM, RAG, FAISS, Web search, session history, and rhetorical distortion***

![MRAicon](MRAicon100x100.png)

December 5, 2025

## Executive Summary

MRA is my personal tool for research, designed as a unified point of interaction to access locally-stored documents and remote web content, perform interactive Q&A with optional rhetorical distortion, and recall and resume sessions from the past. All LLM operations run locally via Ollama and TwistedPair, augmented by RAG (Retrieval-Augmented Generation). The system manages 4 specialized FAISS indices with automatic and manual update strategies.

## System Architecture

![MRA_v3 Diagram](MRA_v3_diagram.jpg)

---

![MRA_v3 screenshot](MRA_screenshot.jpg)

---

## MRA Versions

- **v1/v2**: CLI (files in the root directory)
- **v3**: Integrated Web UI (files in the [V3](./V3) directory)

---

## File structure

```
my-research-assistant
└── V3/
    ├── config.py                    # Configuration (loads .env)
    ├── requirements.txt
    ├── .env                         # API keys (git-ignored)
    ├── server.py                    # FastAPI main server
    ├── chat_manager.py              # Chat session management
    ├── retrieval_manager.py         # Unified FAISS search
    ├── web_search.py                # BRAVE + DuckDuckGo
    ├── twistedpair_client.py        # REST client for distortion
    ├── ollama_client.py             # Ollama interface
    ├── auto_indexer.py
    ├── build_runtime_indices.py
    ├── errors.py
    ├── MRA_v3_1_pdf_to_md.py
    ├── MRA_v3_1_sessions_to_md.py
    ├── MRA_v3_1_webcache_to_md.py
    ├── MRA_v3_2_chunk_md.py
    ├── MRA_v3_3_embed_chunks.py
    ├── MRA_v3_4_verify_index.py
    ├── static/
    │   ├── app.jp
    │   ├── style.css
    │   └── index.html
    ├── data/
    ├── faiss_indices/
    └── utils/
        ├── __init__.py
        ├── document_processor.py
        ├── chunker.py
        ├── embedder.py
        ├── faiss_builder.py
        └── test_pipeline.py       
```

---

## Requirements

- Python 3.10+
- Ollama installed and running with at least one open weight model
- TwistedPair V2 running for LLM distortion
- CUDA GPU
- FAISS indices built for local documents and files

```powershell
# In separate terminal, start ollama server
ollama server
```

```powershell
# In separate terminal, start TwistedPair server
cd 
cd ..\TwistedPair\V2
uvicorn server:app --port 8001 --reload
```

```powershell
# In separate terminal, start MRA_v3 server
cd 
cd ..\MRA_v3
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```


