# My Research Assistant (MRA)

Version 2: November 10, 2025

**My Research Assistant (MRA)** is a private, local document search and Q&A system powered by a local LLM. It was built to solve a personal challenge: accessing and understanding over 1,000 academic PDFs accumulated over years of research. MRA enables conversational search and question-answering over a large corpus without using external services or sending data to the cloud. As of Version 2, MRA is a CLI (command line interface) application, running on Linux OS.

## Features

- Conversational interface for querying local documents and previous sessions
- No cost and private interaction with local LLM via Ollama HTTP API
- Markdown logging for session history and post-session query
- Semantic document search using FAISS and Sentence Transformers
- Integrated Web search via Brave Search API for latest information search 
- Modular utilities for PDF conversion, chunking, embedding, and index updates
- CLI-based task-selections for document search, Q&A, summary, general inquiry and web search

## Planned features

- Distributed local index files for optimal user experience

## Project Structure

MyResearchAssistant/
├── main.py                      # Entry point
├── config.py                    # Paths, model settings, token limits, API keys
├── conversation.py              # Interactive loop
├── markdown_logger.py           # Markdown session logs
├── llm_interface.py             # Local LLM via Ollama API
├── prompt_builder.py            # Prompt assembly
├── retrieval.py                 # FAISS + metadata retrieval
├── utils.py                     # Token counting, history
├── web_search.py                # Global web search
├── faiss_index                  # Semantic index (must be built by util tools)
├── chunk_metadata.json          # Metadata for index (must be built by util tools)
├── README.md                    # readme info
├── LICENSE                      # MIT license
├── requirements.txt             # required libraries
├── logs/                        # folder for logs (MRAlog_*timestamp*.md)
├── util/                             # folder for index and metadata util tools
├── util/pdfs_in/                     # folder Source PDFs
├── util/pdfs_md/                     # folder Markdown-converted PDFs
├── util/pdfs_ocr/                    # folder for OCR-needed PDFs
├── util/MRA_1pdf2text.py             # Converts PDF to text/markdown
├── util/MRA_2chunk_md.py             # Chunks markdown files
├── util/MRA_3embed_chunks_B.py       # Batch embedding
├── util/MRA_3embed_chunks_S.py       # Streaming embedding
├── util/MRA_4query_LLM.py            # Query test
├── util/MRA_6addchunk_md_enhanced.py # Chunks new files in logs folder
├── util/MRA_7update_index_batch.py   # Updates index
├── util/migrate_chunk_metadata.py    # Converts old chunk format
├── util/chunk_md.jsonl               # Output from chunking for embedding
├── util/addchunk_md.jsonl            # Output from index updates
└── util/log_chunk_md.txt             # Chunking log

## Requirements

beautifulsoup4==4.14.2
faiss-cpu==1.12.0
faiss-gpu==1.7.2
fitz==0.0.1.dev2
numpy==2.3.4
pymupdf4llm==0.1.8
requests==2.32.5
sentence_transformers==5.1.2
tiktoken==0.12.0
torch==2.9.0+cpu
tqdm==4.67.1
transformers==4.57.1
trafilatura==2.0.0

In addition, ollama and model files are required

## Setup

### 1. Install dependencies

pip install -r requirements.txt

Or manually:

pip install beautifulsoup4 faiss-gpu fitz numpy pymupdf4llm requests \
sentence_transformers tiktoken torch tqdm transformers

### 2. Install Ollama and pull your local model

url -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull mistral
ollama run mistral

### 3. Prepare your PDFs

- Place original PDFs in `pdfs_in/`
- Run `MRA_1pdf2text.py` to convert them to markdown, outputs in 'pdfs_md' folder
- Run `MRA_2chunk_md.py` to chunk md files and generates chunks_md.jsonl
- Run `MRA_3embed_chunks_B.py` to generate FAISS_index and chunk_metadata.json
- Use `MRA_7update_index_batch.py` to update the FAISS index and metadata

## Usage

Launch the assistant:

python main.py

**Example session:**

🧠 Welcome to My Research Assistant (MRA)
💬 Type 'exit' to quit at any time.

Choose your next task:
[1] General knowledge (@general)
[2] Search documents (@search)
Enter number or keyword: 1
Enter your query or topic: hello

[MRA] (General)
Hello! How can I assist you today? If you have a specific research topic in mind, feel free to ask and I'll do my best to help.

🔁 Ready for your next task.

Choose your next task:
[1] General knowledge (@general)
[2] Search documents (@search)
Enter number or keyword: 2
Enter your query or topic: are there examples of engram being implemented in robots?

Top documents:
[1] ...
[2] ...
.
.
[20] ...

🔁 Ready for your next task.

Choose your next task:
[1] Answer using retrieved documents (@answer)
[2] Summarize a document (@summary)
[3] Search new documents (@search)
[4] General knowledge (@general)
Enter number or keyword:

## Notes

- All processing is local. No cloud dependencies.
- Markdown logs are saved in `logs/` with timestamped filenames.
- You can switch between batch and streaming embedding modes.


