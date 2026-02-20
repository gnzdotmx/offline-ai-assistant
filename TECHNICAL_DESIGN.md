# Technical Explanation of the Offline AI Assistant

This document explains the technical design of the **Offline AI Desktop Assistant**, covering the purpose of each component, the tools used, and how they interact.

## Flow Diagram

![Flow Diagram](imgs/flow_diagram.png)

---
## Project Structure

The application uses a layered package layout: **config**, **core**, **data**, and **llm**. Root-level modules (`rag.py`, `chunker.py`, `embedder.py`, `extractor.py`, `model_manager.py`, `vectorstore.py`, `llm.py`) are compatibility shims that re-export from the packages below so existing imports keep working.

```
offline_ai_assistant/              # Project root
├── offline_ai_assistant/           # Main package
│   ├── __init__.py                # Public API (Config, RAGPipeline, data/llm exports)
│   ├── config/                    # Configuration
│   │   ├── __init__.py
│   │   ├── loading.py             # Config class, load/save, setup_logging
│   │   ├── paths.py               # resolve_under, SafePathResolver (path safety)
│   │   └── schema.py              # validate_settings, get_default_settings, CONFIG_BOUNDS
│   ├── core/                      # Domain and orchestration
│   │   ├── __init__.py
│   │   ├── models.py              # RAGResult, ProcessingResult, TextChunk, GenerationConfig
│   │   ├── interfaces.py         # IEmbedder, IVectorStore, ILLM, IExtractor, IChunker
│   │   ├── rag.py                 # RAGPipeline, create_rag_pipeline, retrieval helpers
│   │   └── rerank.py              # Optional keyword-overlap re-ranking
│   ├── data/                      # I/O and storage
│   │   ├── __init__.py
│   │   ├── extractor.py           # DocumentExtractor (PDF/DOCX), extract_document
│   │   ├── chunker.py             # TextChunker, chunk_document (tiktoken / word fallback)
│   │   ├── embedder.py            # TextEmbedder, EmbeddingCache, create_embedder
│   │   ├── vectorstore.py         # VectorStore (FAISS + SQLite), create_vector_store
│   │   └── model_manager.py       # ModelManager, ModelInfo (download/delete GGUF)
│   ├── llm/                       # Local LLM
│   │   ├── __init__.py            # LocalLLM, LLMManager, create_llm
│   │   └── local_llm.py           # Llama wrapper, prompts, streaming, tokenize/truncate
│   ├── app_ui.py                  # PySide6 GUI (main window, workers, settings, model manager)
│   ├── rag.py                     # Shim → core.rag, core.models
│   ├── chunker.py                 # Shim → data.chunker, core.models (TextChunk)
│   ├── embedder.py                # Shim → data.embedder
│   ├── extractor.py               # Shim → data.extractor
│   ├── vectorstore.py             # Shim → data.vectorstore
│   ├── model_manager.py           # Shim → data.model_manager
│   └── llm.py                     # Shim (shadowed by llm/ package) → llm, core.models
├── requirements.txt
├── Makefile                       # install, run, venv, clean
├── README.md
└── TECHNICAL_DESIGN.md            # This document

# User data (outside repo)
~/.config/ai-offline-assistant/
├── db/                            # FAISS index + SQLite metadata
│   ├── faiss_index
│   └── metadata.db
├── docs/                          # Uploaded documents (copies)
├── models/                        # GGUF files + sentence-transformers cache
├── logs/
│   └── app.log
└── config.json                    # Persisted settings
```



---

## 1. Desktop Application Framework
- **Tool:** `PySide6` (Qt for Python)  
- **Purpose:** Provides a **cross-platform GUI** (Windows, macOS, Linux).  
- **How it works:**  
  - Renders windows, menus, buttons, and text fields natively.  
  - Connects user actions (e.g., "Upload PDF") to backend functions via Qt's **signal-slot mechanism**.  
- **Why chosen:**  
  - Native look-and-feel.  
  - Lighter and faster than Electron.  
  - Cross-platform compatibility.

---

## 2. Document Parsing
- **Tools:**  
  - `pymupdf` → PDF parsing.  
  - `python-docx` → DOCX parsing.  
- **Purpose:** Convert binary files → clean text.  
- **How it works:**  
  - `pymupdf` extracts text page by page from PDFs.  
  - `python-docx` reads Office Open XML and extracts structured text.  
- **Why chosen:**  
  - Lightweight, accurate, and works offline.

---

## 3. Text Chunking
- **Tools:** `tiktoken` (primary), word-based fallback when tiktoken is missing or encoding fails.  
- **Purpose:** Break large documents into **manageable chunks**.  
- **How it works:**  
  - Splits by **token count** (chunk_size, chunk_overlap from config).  
  - Supports structure-preserving mode (paragraph/sentence boundaries) or simple word-based splitting.  
  - When tiktoken is unavailable, uses word count with a configurable ratio so effective chunk size stays conservative.  
- **Why chosen:**  
  - Keeps chunks within the LLM context window.  
  - Improves retrieval accuracy; fallback ensures operation without tiktoken.

---

## 4. Embeddings
- **Tool:** `sentence-transformers` (`all-MiniLM-L6-v2`)  
- **Purpose:** Convert text → **vector embeddings** (high-dimensional arrays).  
- **How it works:**  
  - Transformer model maps semantically similar sentences close in vector space.  
  - Example: “Company revenue rose” ≈ “Business income increased”.  
- **Why chosen:**  
  - Runs offline.  
  - Fast and lightweight (384 dimensions).  
  - Strong accuracy vs. model size.

---

## 5. Vector Database
- **Tools:**  
  - `FAISS` → fast similarity search.  
  - `SQLite` → metadata storage.  
- **Purpose:** Store and search embeddings efficiently.  
- **How it works:**  
  - FAISS indexes vectors for nearest-neighbor search.  
  - SQLite stores metadata (doc name, page, chunk ID).  
- **Why chosen:**  
  - FAISS = scalable + fast.  
  - SQLite = lightweight, file-based, no server needed.

---

## 6. Local LLM
- **Tool:** `llama-cpp-python` (bindings for `llama.cpp`)  
- **Purpose:** Run a **local large language model** (LLM).  
- **How it works:**  
  - Loads quantized GGUF model (e.g., LLaMA 3, Mistral).  
  - Runs inference on CPU/GPU → predicts tokens sequentially.  
  - Supports streaming responses to the UI.  
- **Why chosen:**  
  - 100% offline.  
  - Works on consumer hardware with quantization.  
  - Easy Python integration.
- **Prompt processing (n_batch):** The `n_batch` parameter controls how many tokens are processed at once when feeding the prompt to the model. A larger value (e.g. 1024) can speed up long-prompt processing on capable hardware but increases memory use; default is 512. Configurable via Settings (Advanced) or `llm_n_batch` in config / `OFFLINE_AI_LLM_N_BATCH` env.

---

## 7. Retrieval-Augmented Generation (RAG)
- **Purpose:** Ground LLM answers in **user documents**.  
- **How it works:**  
  1. User query → embedded.  
  2. Vector search returns top-k (or more if re-ranking); optional **per-document cap** (`rag_max_chunks_per_doc`) diversifies sources.  
  3. Optional **re-ranking** (keyword overlap) refines the list before building context.  
  4. Context can be **ordered by score** (retrieval order) or **by document** (chunks grouped by document, then by chunk index).  
  5. RAG prompt built from template (default or chosen template); prompt truncated to fit context window.  
  6. LLM generates answer (non-streaming or **streaming** tokens to the UI).  
  7. Response includes sources (file, score, preview).  
- **Extra capabilities:**  
  - **Streaming:** `query_stream()` yields status, sources, token-by-token text, and final timing.  
  - **Content generation:** `generate_content()` supports a context query (retrieval + prompt) or a direct prompt.  
- **Why chosen:**  
  - Reduces hallucinations; answers cite user data.  
  - Re-ranking and document-order options improve relevance and coherence.

---

## 8. Drafting Templates
- **Purpose:** Provide **structured output** for professional use cases.  
- **How it works:**  
  - Predefined prompts (e.g., project plan, summary, risk report).  
  - Inserts user documents + context.  
  - Passes prompt to LLM → generates formatted content.  
- **Why chosen:**  
  - Ensures consistent, reusable outputs.  
  - Saves time for client-facing reports.

---

## 9. Data Privacy
- **Design choices:**  
  - No external API calls (no OpenAI, no cloud).  
  - All models, embeddings, and docs stored locally in `~/.config/ai-offline-assistant/`.  
  - SQLite database can be encrypted (e.g., with `sqlcipher`).  
  - User data separated from application code for security and portability.  
- **Why chosen:**  
  - Meets compliance and client confidentiality requirements.  
  - Works in **air-gapped** environments.  
  - Follows standard user data storage conventions.

---

## 10. Configuration Management
- **Components:**  
  - **Schema** (`config/schema.py`): Validates and clamps all settings (chunk_size, top_k, LLM params, embedding model, rag_rerank, etc.); returns validated dict and list of warnings.  
  - **Paths** (`config/paths.py`): Resolves paths under allowed base dirs; prevents path traversal.  
  - **Loading** (`config/loading.py`): `Config` class, load/save from JSON, env overrides, `setup_logging()`.  
- **How it works:**  
  - Settings in `~/.config/ai-offline-assistant/config.json`; loaded on startup.  
  - UI and code use `Config` attributes; save writes validated values.  
  - Environment variables (e.g. `OFFLINE_AI_MODELS_DIR`, `OFFLINE_AI_EMBEDDING_BATCH_SIZE`) override defaults.  
- **Why chosen:**  
  - Single source of truth with validation and safe paths.  
  - XDG-compliant; easy to backup and migrate.

---

## 11. Testing
- **Scope:** Unit tests for config (loading, paths, schema), core (RAG pipeline, rerank, models), data (extractor, chunker, embedder, vector store, model manager), and LLM (local_llm, LLMManager) using mocks so no real models or network are required. Root shim modules have small tests that assert re-exports.  
- **Running:** `pytest offline_ai_assistant/ -v`; add `--cov=offline_ai_assistant --cov-report=term-missing` for coverage.  
- **GUI tests** (`app_ui_test.py`): Skipped by default; set `OFFLINE_AI_RUN_GUI_TESTS=1` to run worker and widget tests (requires PySide6 and display or offscreen).

---

## Workflow Overview
1. User uploads a PDF/DOCX (copied to `~/.config/ai-offline-assistant/docs/`).  
2. Text extracted → chunked → embedded.  
3. Embeddings stored in FAISS + SQLite (in `~/.config/ai-offline-assistant/db/`).  
4. User submits query.  
5. Query embedded → FAISS retrieves top-k chunks.  
6. RAG pipeline builds prompt with query + context.  
7. LLM generates grounded response.  
8. Response + citations displayed in GUI.  
9. Optional: drafting templates generate structured reports.  

---



## Summary of Tools

| Tool                     | Purpose                     | Why Chosen                        |
|--------------------------|-----------------------------|-----------------------------------|
| **PySide6**              | GUI                         | Native, cross-platform            |
| **pymupdf, python-docx** | Document parsing            | Reliable, offline                 |
| **tiktoken**             | Text chunking (primary)     | Token-aware, LLM-friendly        |
| **sentence-transformers** | Embeddings                | Fast, lightweight, offline        |
| **FAISS + SQLite**       | Vector DB + metadata        | Scalable, lightweight, local     |
| **llama-cpp-python**     | Local LLM runtime           | Offline, efficient               |
| **RAG pipeline**         | Grounded AI, streaming      | Reduces hallucinations, citations |
| **core.interfaces**      | IEmbedder, IVectorStore, etc. | Pluggable components            |
| **config (schema, paths, loading)** | Validation, paths, persistence | Safe, XDG-compliant config   |
| **pytest**               | Unit and shim tests         | Coverage without real models     |

---