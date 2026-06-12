# 📈 Financial & Academic RAG Chatbot

An advanced, production-grade agentic chatbot built with **LangGraph**, **FastAPI**, and a state-of-the-art **Hybrid RAG** (Retrieval-Augmented Generation) pipeline. This application is optimized for retrieving and analyzing complex financial statements (such as annual reports, tables, spreadsheets) and dense academic publications (like the GPT-3 paper).

---

## 🚀 Key Features

### 🧠 Agentic Architecture (LangGraph)
* **Conditional State Machine**: Dynamic routing between standard conversation, knowledge-base querying, python sandbox execution, and web navigation.
* **Stop Generation**: Immediate client-side streaming cancellation via `asyncio` task registers.
* **Long-Term Memory**: Automatic context summarization of older dialogue steps to prevent token bloat while maintaining state history.
* **Thread Persistence**: Powered by a robust `PostgresSaver` checkpointer ensuring user conversations survive server restarts.

### 🔍 Advanced Hybrid RAG Pipeline
* **Multi-Format Ingestor**: Layout-aware table extraction from PDFs (PyMuPDF/Fitz), Excel spreadsheets (`.xlsx`/`.xls`/`.csv`), and Word files (`.docx`).
* **Dense Embedding Retriever**: ChromaDB utilizing the high-performance **`BAAI/bge-base-en-v1.5`** embedding model.
* **Sparse Keyword Search**: Customized **BM25 retrieval** with layout-aware filters (automatically excludes bibliography indexes and header/footer noise).
* **Reranking & Fusion**: Employs **Reciprocal Rank Fusion (RRF)** to merge dense and sparse inputs, followed by a **CrossEncoder Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for selecting premium candidate blocks.

### 🛠️ Agent Tool Suite
* **Python Interpreter**: Isolated code sandbox for solving mathematical tasks, analyzing tables, and computing percentages.
* **Web Navigation (Playwright)**: Full browser client supporting site navigation, form inputs, button clicks, and screenshot uploads.
* **Web Search**: Broad Google-based queries for real-time information retrieval.
* **Financial Stocks API**: Fetch real-time market data and stock details.

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

### Prerequisites
* Anaconda / Miniconda installed.
* PostgreSQL database instance.
* API Keys for Google Gemini (and external tools if desired).

### 1. Environment Setup
Clone the repository and initialize the Conda environment:
```bash
conda env create -f llm.yml
conda activate llm
```

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key
DB_URI="postgresql://username:password@localhost:5432/dbname"
```

### 3. Run the Backend Server
Start the FastAPI server via Uvicorn:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open your browser and navigate to `http://localhost:8000` to interact with the application.

---

## 📝 Folder Structure

```text
├── app/
│   ├── agent/
│   │   ├── graph.py       # LangGraph state machine & router
│   │   └── tools.py       # Tool registry (Web, Python, Playwright, Stocks)
│   ├── core/
│   │   ├── config.py      # App configurations & secrets
│   │   └── security.py    # Input validation utilities
│   ├── services/
│   │   ├── audio.py       # Whisper audio transcription services
│   │   └── rag_pipeline.py# Advanced Hybrid RAG, BM25, and Reranking logic
│   └── main.py            # FastAPI endpoints & Lifespan startup hooks
├── static/
│   ├── app.js             # Frontend reactive interface logic
│   ├── index.html         # Main dashboard layout
│   └── styles.css         # Glassmorphic dark styling system
├── llm.yml                # Conda environment configuration
└── README.md              # Project documentation
```

---

## 🔗 GitHub Repository
Find the active repository online at: [https://github.com/HARSH-GOHIL-git/financial-chatbot](https://github.com/HARSH-GOHIL-git/financial-chatbot)
