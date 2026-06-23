# 📈 Financial & Academic RAG Chatbot

An advanced, production-grade agentic chatbot built with **LangGraph**, **FastAPI**, and a state-of-the-art **Hybrid RAG** (Retrieval-Augmented Generation) pipeline. This application is optimized for retrieving and analyzing complex financial statements (such as annual reports, tables, spreadsheets) and dense academic publications (like the GPT-3 paper).

---

## 🚀 Key Features

### 🧠 Agentic Architecture (LangGraph)
* **Conditional State Machine**: Dynamic routing between standard conversation, knowledge-base querying, python sandbox execution, and web navigation.
* **Stop Generation**: Immediate client-side streaming cancellation via `asyncio` task registers.
* **Long-Term Memory**: Automatic context summarization of older dialogue steps to prevent token bloat while maintaining state history.
* **Thread Persistence**: Powered by a robust `PostgresSaver` checkpointer ensuring user conversations survive server restarts.
* **Thread Isolation**: Automatic validation of conversation thread IDs against the local database to ensure session isolation, preventing cross-talk between user threads.

### 🔍 Advanced Hybrid RAG Pipeline
* **Multi-Format Ingestor**: Layout-aware table extraction from PDFs (PyMuPDF/Fitz), Excel spreadsheets (`.xlsx`/`.xls`/`.csv`), and Word files (`.docx`).
* **Dense Embedding Retriever**: ChromaDB utilizing the high-performance **`BAAI/bge-base-en-v1.5`** embedding model.
* **Sparse Keyword Search**: Customized **BM25 retrieval** with layout-aware filters (automatically excludes bibliography indexes and header/footer noise). BM25 indices are cached in RAM on-the-fly and stored only for current active documents.
* **Reranking & Fusion**: Employs **Reciprocal Rank Fusion (RRF)** to merge dense and sparse inputs, followed by a **CrossEncoder Reranker** (`BAAI/bge-reranker-base`) for selecting premium candidate blocks.
* **Visual Document Ingestion & Caching**: Layout scanning via **RapidOCR** during document ingestion. When visual chunks are retrieved, the pipeline lazily calls **Gemini Flash VLM** to construct detailed descriptions, caching them in SQLite via SHA-256 hashes to prevent redundant API hits and conserve VLM tokens.

### 🛡️ Production Hardening & Security
* **No Client Leakage**: Replaces raw stack trace outputs on endpoints with generic error messages. Detailed traces are logged server-side only.
* **Strict Size Enforcements**: Prevents memory exhaustion with strict payload limits (50MB max for PDF/Excel uploads, 25MB max for Whisper voice recordings).
* **Path Sandboxing**: Validates all filesystem operations against directory traversal (`..` checks) and restricts access to sensitive folders (ignoring `.env`, `.git`, databases, and application code).
* **Python REPL AST Checking**: Evaluates execution inputs using Abstract Syntax Trees, blocking dangerous libraries (`os`, `subprocess`, `socket`, `requests`), built-ins (`eval`, `exec`, `open`), and file manipulation commands.
* **Output Sanitizer**: Automatically cleanses database references, configuration names, and system files from all tool responses before showing them to the user.

### 🛠️ Agent Tool Suite
* **Python Interpreter**: Isolated code sandbox for solving mathematical tasks, analyzing tables, and computing percentages.
* **Web Navigation (Playwright)**: Full browser client supporting site navigation, form inputs, button clicks, and screenshot uploads. The AI intelligently utilizes web search before executing exact Playwright browser actions.
* **Web Search**: Broad Google-based queries (via DuckDuckGo Search) for real-time information retrieval.
* **Financial Stocks API**: Fetch real-time market data and stock details (via AlphaVantage).
* **Filesystem Access**: Secure, sandboxed read/write tools.

### 📊 Centralized Observability & Logging
* **Structured Rotating Logger**: Console output stream paired with a rotating file logger (`logs/app.log`) capping logfiles at 10MB each (retains up to 5 backups).
* **Telemetry & Tracing**: Direct support for LangSmith tracing, logging rate-limit warnings, fallback switches, thread cancellations, and RAG execution timings.

### 🎙️ Audio Transcription
* Built-in **Whisper model** processor on the backend for transcribing voice messages in real time.

### 💎 Premium User Interface
* Stunning modern **Glassmorphism dark UI** with real-time markdown rendering, syntax-highlighted code blocks (Prism.js), collapsible nested tool-execution logs, and custom audio recording waves.

---

## 🗺️ System Architecture

```mermaid
graph TD
    UI[Frontend: Glassmorphic Web UI] <-->|Server-Sent Events / JSON| API[FastAPI Backend]
    API <-->|Thread Checkpoint State| Postgres[(PostgreSQL DB)]
    API -->|Prompt & State| Graph{LangGraph State Machine}
    
    Graph -->|Route| ChatNode[chat_node: Router & LLM]
    Graph -->|Route| ToolNode[tools_node: Execution Sandbox]
    
    ToolNode -->|Invoke RAG| RAG[Hybrid RAG Engine]
    ToolNode -->|Run Scripts| Python[Python Sandbox Interpreter]
    ToolNode -->|Navigate Web| Playwright[Playwright Browser MCP]
    ToolNode -->|Search| WebSearch[Google Web Search API]
    
    RAG -->|Vector Search| Chroma[(ChromaDB Vector Store)]
    RAG -->|Sparse Search| BM25[BM25 Indexer]
    RAG -->|Merge| RRF[Reciprocal Rank Fusion]
    RRF -->|Re-Rank| CrossEncoder[BGE Cross-Encoder Reranker]
```

---

## 🛠️ Setup & Installation

### 📋 Prerequisites Checklist
Ensure you have the following software installed on your system before proceeding:
1. **Miniconda / Anaconda** (Python 3.10+)
2. **PostgreSQL** (v14+; used for LangGraph session checkpointers and thread files metadata)
3. **Node.js & NPM** (v18+; required for executing Model Context Protocol filesystem and Playwright browser servers)
4. **LibreOffice** (required in headless mode to convert uploaded Word `.docx` documents to PDF)
5. **FFmpeg** (required by the audio transcription service `faster-whisper` to parse voice uploads)
6. **Google Gemini API Key** (used for primary LLM nodes and VLM image analysis)

---

### 📥 Step-by-Step Installation

#### Step 1: Clone the Repository & Navigate to Workspace
```bash
git clone https://github.com/HARSH-GOHIL-git/financial-chatbot.git
cd financial-chatbot/chatbot-17
```

#### Step 2: Set Up the Conda Environment
Use the provided `llm.yml` or `conda-requirements.txt` to restore the environment:

##### Option A: Using Conda Environment File (Recommended)
```bash
conda env create -f llm.yml
conda activate llm
```

##### Option B: Manual Installation (Fallback for other Platforms/OS)
If the exact build versions in the requirements file fail to resolve on your platform, create a clean environment and install manually:
```bash
conda create --name llm python=3.10 -y
conda activate llm
pip install fastapi uvicorn psycopg rank-bm25 sentence-transformers llama-cpp-python faster-whisper Pillow google-genai langchain-core langchain-google-genai langgraph langchain-community langchain-text-splitters python-dotenv huggingface_hub starlette starlette-sse openpyxl pandas requests ddgs playwright langchain-mcp-adapters rapidocr-onnxruntime
```

#### Step 3: Install System Dependencies
Install system libraries for document conversion (`libreoffice`) and voice processing (`ffmpeg`):
* **Ubuntu / Debian**:
  ```bash
  sudo apt update
  sudo apt install libreoffice ffmpeg -y
  ```
* **macOS (Homebrew)**:
  ```bash
  brew install libreoffice ffmpeg
  ```

#### Step 4: Install Playwright & Browsers
To enable web crawling tools, set up Playwright within the active environment:
```bash
playwright install --with-deps
```

#### Step 5: Setup the PostgreSQL Database
1. Connect to your local PostgreSQL instance:
   ```bash
   sudo -u postgres psql
   ```
2. Create the target database, user, and permissions:
   ```sql
   CREATE DATABASE chatbot_db;
   CREATE USER harsh WITH PASSWORD 'harsh';
   GRANT ALL PRIVILEGES ON DATABASE chatbot_db TO harsh;
   ```

#### Step 6: Configure Environment Variables
Create a `.env` file in the root folder of the project (`chatbot-17/`) containing the following variables:
```env
# Google Gemini API Credentials (Required)
GOOGLE_API_KEY="AIzaSyYourRealGeminiKeyHere"

# PostgreSQL connection URI (Required)
DB_URI="postgresql://harsh:harsh@localhost:5432/chatbot_db"

# Sandboxed root directory for agent tool outputs
MCP_FS_ROOT="/home/neuramonks/Desktop/Harsh/LLM/LangGraph/langgraph-practice/chatbot-17"

# AlphaVantage API Token for live stock prices (Optional)
ALPHA_VANTAGE_KEY="your_alphavantage_api_key_here"

# LangSmith Telemetry (Optional - for developer debugging)
LANGSMITH_TRACING="false"
LANGSMITH_API_KEY=""
LANGSMITH_PROJECT=""
```

#### Step 7: Launch the Chatbot Backend
Start the FastAPI server via Uvicorn from the project root:
```bash
python -m uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

#### Step 8: Validate the Setup
* **Chat Dashboard UI**: Open [http://localhost:8000](http://localhost:8000) in your web browser.
* **Health Endpoint**: View the diagnostic check at [http://localhost:8000/health](http://localhost:8000/health).
* **Tools Registry Debugger**: Check active MCP/Agent tools status at [http://localhost:8000/debug/tools](http://localhost:8000/debug/tools).

---

## 📝 Folder Structure

```text
├── app/
│   ├── agent/
│   │   ├── graph.py          # LangGraph state machine & router
│   │   └── tools.py          # Tool registry (Web, Python, Playwright, Stocks)
│   ├── core/
│   │   ├── config.py         # App configurations & secrets
│   │   ├── logger.py         # Centralized rotating file logger configuration
│   │   └── security.py       # Input validation & security tools
│   ├── services/
│   │   ├── audio.py          # Whisper audio transcription services
│   │   └── rag_pipeline.py   # Hybrid RAG, BM25, and Reranking logic
│   └── main.py               # FastAPI endpoints & Lifespan startup hooks
├── static/
│   ├── app.js                # Frontend reactive interface logic
│   ├── index.html            # Main dashboard layout
│   └── styles.css            # Glassmorphic dark styling system
├── logs/                     # Ignored logs folder (holds app.log)
├── chroma_db/                # Local Chroma vector database
├── image_cache.db            # SQLite cache of VLM descriptions
├── DOCUMENTATION.md          # Full Technical Reference Manual
├── conda-requirements.txt   # Conda packages list (Recommended setup)
├── llm.yml                   # Conda environment configuration
└── README.md                 # Project documentation
```

---

## 🔗 GitHub Repository
Find the active repository online at: [https://github.com/HARSH-GOHIL-git/financial-chatbot](https://github.com/HARSH-GOHIL-git/financial-chatbot)
