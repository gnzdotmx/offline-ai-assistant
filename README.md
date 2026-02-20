# Offline AI Assistant

A fully offline desktop AI assistant that provides document analysis and question-answering capabilities with complete privacy and security. No external API calls, all processing happens locally on your machine.

## Demo Video

[![Watch Demo Video](https://img.shields.io/badge/▶️%20Watch%20Demo%20Video-FF6B6B?style=for-the-badge&logo=vimeo&logoColor=white)](https://vimeo.com/1128449188)

> **Click the button above to watch the demo video showing how the Offline AI Assistant works**


## Features

- **Document Processing**: Upload and parse PDF and DOCX files
- **Semantic Search**: Find relevant information across all your documents
- **AI-Powered Responses**: Get intelligent answers with source citations
- **Complete Privacy**: All data stays on your local machine
- **Efficient**: Uses quantized models for fast CPU inference
- **Modern UI**: Clean, intuitive PySide6 interface
- **No internet connection required** after setup
- **All data stays local** - documents, embeddings, responses
- **No telemetry or tracking**


## Requirements

### System Requirements

- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space (more for models and documents)
- **CPU**: Modern multi-core processor (GPU optional but not required)

### Software Requirements

- Python 3.10 or higher
- pip (Python package manager)

## Installation

### Option A: Using Make (recommended)

From the project root:

```bash
# Clone (first time only)
git clone https://github.com/gnzdotmx/offline-ai-assistant.git
cd offline-ai-assistant

# Create virtualenv and install dependencies
make install

# Run the application
make run
```

To use an **existing models folder** (e.g. shared with other projects) instead of the default `~/.config/ai-offline-assistant/models`:

```bash
make run MODELS_DIR=/path/to/your/models
# or set once:
export OFFLINE_AI_MODELS_DIR=/path/to/your/models
make run
```

Other targets: `make help` (list targets), `make venv` (only create venv), `make clean` (remove venv and cache).

### Option B: Manual setup

#### 1. Clone the Repository

```bash
git clone https://github.com/gnzdotmx/offline-ai-assistant.git
cd offline-ai-assistant
```

#### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\\Scripts\\activate

# On macOS/Linux
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run the Application

```bash
python -m offline_ai_assistant.app_ui
```

## Usage Guide

### First Launch

1. **Launch the application**
2. **Download an AI model**:
   - Go to Edit → Manage Models
   - Select a model from the "Available Models" tab
   - Click "Download Selected Model"
   - Wait for the download to complete (models are stored in ~/.config/ai-offline-assistant/models/)
   - Switch to "Installed Models" tab and click "Use This Model"
   
   Or manually configure:
   - Go to Edit → Settings
   - Click "Manage Models..." or browse to your GGUF model file
   - Adjust other settings as needed
   - Click Save

### Uploading Documents

1. **Click "Upload Documents"** or use File → Upload Documents
2. **Browse and select PDF or DOCX files** from anywhere on your computer
3. **Wait for processing** - documents will be:
   - Copied to `~/.config/ai-offline-assistant/docs/` for management
   - Text extracted and cleaned
   - Split into chunks
   - Embedded using AI
   - Stored in local vector database
   
Note: Your original files remain untouched in their original location. The application works with copies stored in the managed directory.

### Asking Questions

1. **Type your question** in the input box at the bottom
2. **Select a template** (optional):
   - **Default**: General question answering
   - **Summary**: Document summarization
   - **Project Plan**: Generate project plans
   - **Executive Briefing**: Create executive summaries
3. **Click Send** or press Enter
4. **View the response** with source citations

### Managing Documents

- **View all documents** in the left panel
- **Delete documents** by selecting and clicking Delete
- **Refresh the list** with the Refresh button
- **View statistics** via View → Show Statistics

## Configuration

### Settings Panel

Access via Edit → Settings:

- **Chunk Size**: Number of tokens per text chunk (default: 512)
- **Chunk Overlap**: Overlap between chunks (default: 50)
- **Top-K Results**: Number of chunks to retrieve (default: 5)
- **Max Tokens**: Maximum tokens to generate (default: 1024)
- **Temperature**: Creativity level 0.0-2.0 (default: 0.7)
- **Top-P**: Nucleus sampling parameter (default: 0.9)
- **GPU Layers**: Number of layers on GPU (default: 0 for CPU-only)

**Embedding model:** The default is `all-MiniLM-L6-v2` (fast, 384 dims). You can switch to a stronger model (e.g. `all-mpnet-base-v2`) in Edit → Settings for better retrieval on harder queries. Changing the embedding model requires re-processing documents: the app will warn if the index was built with a different model and can clear the index so you can re-upload.

**Configuration persistence:**
- Settings are saved when you click "Save"
- Stored in `~/.config/ai-offline-assistant/config.json`
- Loaded automatically on startup
- Use "Reset to Defaults" to restore factory settings

### File Locations

All user data is stored in `~/.config/ai-offline-assistant/`:

- **Documents**: `docs/` - Uploaded document files
- **Database**: `db/` - Vector index and metadata
- **Models**: `models/` - AI model files
- **Logs**: `logs/` - Application logs

### Model Management

The application includes a built-in model manager accessible via Edit → Manage Models:

**Available Features:**
- Browse available models for download
- View model details (size, quantization, context length)
- Download models directly from Hugging Face
- View and manage installed models
- Delete models to free disk space
- Quick-select models for use

**Supported Models:**
- Llama 2 7B Chat (Q4_K_M) - 3.8 GB
- Mistral 7B Instruct (Q4_K_M) - 4.1 GB
- Code Llama 7B Instruct (Q4_K_M) - 3.8 GB
- Phi-2 (Q4_K_M) - 1.6 GB (tiny but capable)

## Advanced Configuration

### GPU Acceleration

To enable GPU acceleration (if you have a compatible NVIDIA GPU):

1. **Install CUDA toolkit** (11.8 or 12.x)
2. **Install GPU dependencies**:
   ```bash
   pip uninstall llama-cpp-python
   pip install llama-cpp-python --force-reinstall --no-cache-dir --config-settings cmake.args=-DLLAMA_CUBLAS=on
   ```
3. **Update settings** to use GPU layers (start with 20-30)

### Custom Models

The application supports any GGUF-format model. To use a custom model not in the catalog:

1. **Download a compatible GGUF model** from Hugging Face or other sources
2. **Place it in `~/.config/ai-offline-assistant/models/`**
3. **Update the model path in Settings** (Edit → Settings → Browse)

### Retrieval re-ranking (optional)

For better answer relevance, you can enable a second-stage re-ranking of search results:

- **Off by default**: No extra dependencies or latency.
- **When enabled**: The app retrieves more candidates (e.g. 3× your Top-K), re-ranks them by keyword overlap with the query, then keeps the top-K for the prompt. This can surface more relevant chunks than vector similarity alone.

**Config (in `config.json` or via settings if exposed in UI):**

- **`rag_rerank`** (boolean, default: `false`) – Set to `true` to enable re-ranking.
- **`rag_rerank_candidate_multiplier`** (integer, 2–5, default: `3`) – Retrieve `top_k × multiplier` candidates, then re-rank down to `top_k`.

**Dependencies:** The built-in re-ranker uses only the standard library (keyword/overlap scoring). No extra packages (e.g. cross-encoders) are required. For even better relevance you could add an optional cross-encoder (e.g. `sentence-transformers` cross-encoder) in the future; that would be an extra dependency.

### Environment Variables

You can override default settings with environment variables:

- **`OFFLINE_AI_MODELS_DIR`** – Custom folder for LLM and embedding models. Use this if you already keep GGUF models (and optionally sentence-transformers cache) in another project or shared location. Default: `~/.config/ai-offline-assistant/models`. Example: `export OFFLINE_AI_MODELS_DIR=/shared/ai-models`
- **`OFFLINE_AI_MODEL_PATH`** – Path to a specific GGUF model file (overrides path in settings when supported).
- **`OFFLINE_AI_CHUNK_SIZE`** – Override chunk size (e.g. 1024).
- **`OFFLINE_AI_GPU_LAYERS`** – Override GPU layers (e.g. 20).
- **`OFFLINE_AI_EMBEDDING_BATCH_SIZE`** – Chunk batch size when embedding documents (1–512). Default: 32. Use a **smaller value** (e.g. 8 or 16) on low-memory machines to reduce RAM use; use a **larger value** for faster processing of large documents.
- **`OFFLINE_AI_EMBEDDING_SHOW_PROGRESS`** – Set to `0`, `false`, or `no` to disable the progress bar during document embedding (e.g. when running headless or in a script). Default: progress bar shown.

## Architecture

- **Language**: Python 3.10+
- **Package layout**: `config/` (validation, paths, loading), `core/` (models, interfaces, RAG pipeline, rerank), `data/` (extractor, chunker, embedder, vector store, model manager), `llm/` (local LLM). Root modules (`rag`, `chunker`, `embedder`, etc.) are compatibility shims re-exporting from these packages.
- **LLM**: llama-cpp-python with GGUF models
- **Embeddings**: sentence-transformers (default all-MiniLM-L6-v2)
- **Vector search**: FAISS + SQLite metadata; optional per-document cap and document-order context
- **Re-ranking**: Optional keyword-overlap re-ranking (off by default; no extra deps)
- **Document parsing**: PyMuPDF (PDF) + python-docx (DOCX)
- **Chunking**: Token-aware with tiktoken (word-based fallback when tiktoken unavailable)
- **GUI**: PySide6 (Qt for Python)

## Testing

From the project root (with dependencies installed):

```bash
# Run all tests
pytest offline_ai_assistant/ -v

# With coverage report
pytest offline_ai_assistant/ --cov=offline_ai_assistant --cov-report=term-missing
```

- **Unit tests** cover core RAG logic, config validation, data layer (chunker, embedder, extractor, vector store, model manager), and LLM layer with mocks. Root shim modules have small compatibility tests.
- **GUI tests** (`app_ui_test.py`) are skipped by default to avoid Qt/display in CI. Set `OFFLINE_AI_RUN_GUI_TESTS=1` to run worker and widget tests when PySide6 and a display (or offscreen) are available.

## Troubleshooting

### Common Issues

**Model not loading:**
- Ensure the model file path is correct
- Check that you have enough RAM
- Try a smaller model (Q4_K_M instead of Q8_0)

**Slow performance:**
- Reduce chunk size and top-k results
- Enable GPU acceleration if available
- Use a smaller/faster model

**Document upload fails:**
- Check file format (PDF/DOCX only)
- Ensure file is not corrupted
- Check available disk space

**Memory errors:**
- Use a smaller model
- Reduce context length in settings
- Close other applications
- **When embedding documents:** set `OFFLINE_AI_EMBEDDING_BATCH_SIZE=8` (or add `"embedding_batch_size": 8` to `config.json`) to use less RAM during indexing.

### Log Files

Check the log files for detailed error information:
- Location: `~/.config/ai-offline-assistant/logs/app.log`
- Contains detailed debugging information

### Performance Tips

1. **Use SSD storage** for better I/O performance
2. **Close unnecessary applications** to free RAM
3. **Use quantized models** (Q4_K_M, Q5_K_M) for speed
4. **Adjust chunk size** based on your documents
5. **Enable GPU** if you have compatible hardware
6. **Prompt processing speed:** On capable hardware, increasing **prompt batch size** (Settings → Advanced) can speed up handling of long prompts; higher values use more memory. Default is 512; try 1024 or 2048 if you have spare VRAM/RAM.

## Technical Design

See [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md) for component descriptions, project structure, data flow, and tool choices.

