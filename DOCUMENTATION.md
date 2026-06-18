# LangGraph-Based Chatbot: Technical Reference Manual

**Version**: 16 (Production-Hardened) | **Last Updated**: 2026-06-18  
**Stack**: FastAPI · LangGraph · ChromaDB · PostgreSQL · RapidOCR · Gemini · Whisper · Playwright MCP

---

## 1. Project Overview

This application is a production-hardened, thread-safe agentic chatbot built on **FastAPI** and **LangGraph**. It provides:

- A glassmorphic dark-mode chat UI with real-time markdown rendering, collapsible tool logs, and syntax-highlighted code blocks.
- High-performance **Hybrid RAG** (BM25 + ChromaDB → RRF → CrossEncoder reranking) supporting PDF, DOCX, Excel, and CSV files with full thread-level isolation.
- **OCR-based image ingestion** (RapidOCR) at index time and **Gemini Flash VLM** for deep image analysis at query time — SHA-256 image deduplication via SQLite cache eliminates redundant API calls.
- A full agent tool suite: Web Search, Python REPL, Stock Ticker, Calculator, Filesystem MCP, and Playwright Browser MCP.
- **Structured rotating file logging** (`logs/app.log`) with severity levels across every module.
- Hardened security: path sandboxing, AST-based Python code safety, and output sanitization.

---

## 2. Project Structure

```
chatbot-16/
├── app/
│   ├── agent/
│   │   ├── graph.py          # LangGraph state machine, LLM fallback, memory summarization
│   │   └── tools.py          # Tool registry: web search, Python REPL, MCP bridge, Playwright RAG
│   ├── core/
│   │   ├── config.py         # .env loader (project-relative path), MCP server schemas
│   │   ├── logger.py         # Centralized logging: RotatingFileHandler + console, get_logger()
│   │   └── security.py       # Path sandboxing, AST code safety, output sanitizer
│   ├── services/
│   │   ├── audio.py          # faster-whisper transcription service
│   │   └── rag_pipeline.py   # Hybrid RAG, ingestion pipeline, RapidOCR, Gemini VLM cache
│   └── main.py               # FastAPI endpoints, lifespan hooks, SSE streaming, upload routes
├── static/
│   ├── index.html            # Main UI layout
│   ├── app.js                # SSE stream parser, session manager, recorder, export handler
│   ├── style.css             # Glassmorphic dark design system
│   └── extracted_images/     # Persistent extracted PDF image files
├── logs/                     # Rotating application log files (excluded from git)
│   └── app.log
├── backend.py                # Root entrypoint → imports app/main.py
├── chroma_db/                # Local ChromaDB vector storage
├── image_cache.db            # SQLite SHA-256 image description cache
├── .env                      # API keys and environment config (never commit)
└── llm.yml                   # Conda environment definition
```

---

## 3. Environment & Configuration

Variables are loaded in `app/core/config.py` using a **project-relative path** — always resolves to `chatbot-16/.env` regardless of which machine or directory uvicorn is launched from.

| Variable | Description | Required |
|---|---|---|
| `GOOGLE_API_KEY` | Google Gemini API key (LLM, VLM, query expansion) | ✅ Required |
| `DB_URI` | PostgreSQL connection string for LangGraph checkpoints | ✅ Required |
| `MCP_FS_ROOT` | Filesystem root the agent is allowed to read/write | Optional (defaults to `~`) |
| `ALPHA_VANTAGE_KEY` | Stock ticker API key for `stock_price` tool | Optional |
| `LANGSMITH_API_KEY` | LangSmith tracing token | Optional |
| `LANGSMITH_TRACING` | Enable/disable LangSmith tracing (`true`/`false`) | Optional |
| `LANGSMITH_PROJECT` | LangSmith project group name | Optional |

---

## 4. Logging System

Logging is configured once at application startup via `app/core/logger.py`.

### Setup
```python
from app.core.logger import get_logger
logger = get_logger(__name__)
```
`setup_logging()` is called inside `lifespan()` in `main.py` before any other initialization.

### Handlers
| Handler | Destination | Rotation |
|---|---|---|
| `StreamHandler` | Terminal stdout | None |
| `RotatingFileHandler` | `logs/app.log` | 10 MB max, 5 backup files |

### Log Format
```
2026-06-18 10:28:05 | INFO     | app.main             | [App] Ready.
2026-06-18 10:28:15 | WARNING  | app.agent.graph      | [LLM] Rate-limit/server error — waiting 6.5s before retry.
2026-06-18 10:28:22 | ERROR    | app.main             | [App] PDF upload failed.
```

### What Gets Logged

| Event | Level | Module |
|---|---|---|
| Server startup / shutdown | INFO | main |
| Model preload (CrossEncoder, Whisper, Embeddings) | INFO | rag_pipeline, audio |
| MCP server connected / failed | INFO / ERROR | main |
| PDF / DOCX / Excel upload received | INFO | main |
| File too large (>50 MB) | WARNING (HTTP 413) | main |
| Ingestion chunks added | INFO | rag_pipeline |
| Image OCR result | DEBUG | rag_pipeline |
| BM25 cache eviction | WARNING | rag_pipeline |
| Query expansion variants | INFO | rag_pipeline |
| Gemini VLM called at query time | INFO | rag_pipeline |
| LLM rate limit / retry | WARNING | graph |
| LLM fallback model switch | WARNING | graph |
| Thread cancelled | INFO | graph |
| Client disconnected from stream | INFO | main |
| DB / vectorstore errors | ERROR | main, graph |
| All unhandled exceptions | ERROR + traceback | all modules |

---

## 5. Architecture & Execution Flow

### A. Ingestion Pipeline

Triggered by `/upload-pdf` (PDF, DOCX) or `/upload-excel` (CSV, Excel).

**File size limits enforced at upload:**
- PDF / DOCX / Excel / CSV: **50 MB max**
- Audio: **25 MB max**

```
[Upload]
   ├── DOCX → LibreOffice headless → temp PDF
   ├── PDF → fitz parser
   └── Excel/CSV → saved to MCP_FS_ROOT (not os.getcwd())

[Page Loop]
   ├── Tables: page.find_tables() → Markdown table → "table" chunk
   ├── Vector Graphics: path count > 10 → page snapshot PNG → "image" chunk
   ├── Images: page.get_images()
   │     ├── Filter: w<150px OR h<150px OR size<20KB OR aspect>3.0 → skip
   │     ├── SHA-256 hash → SQLite cache lookup
   │     │     ├── Hit  → reuse cached description → "image" chunk (vlm_done=True)
   │     │     └── Miss → RapidOCR on image file → OCR text as search hint
   │     │             ├── With surrounding text → "surrounding_text\n[Visual hint: ocr]"
   │     │             └── Without text         → ocr_text only
   │     │             → "image" chunk (needs_vlm=True, vlm_done=False)
   │     └── Saved: static/extracted_images/<filename>/img_page_X_xrefY.png
   └── Text: non-table blocks → RecursiveCharacterTextSplitter(800, 150) → "text" chunks

[Embed & Store]
   └── BAAI/bge-base-en-v1.5 → ChromaDB (filtered by thread_id + filename)
```

**Image ingestion note**: SmolVLM has been removed. RapidOCR runs at index time for text extraction. Deep semantic image analysis (Gemini Flash) runs lazily at query time only when an image chunk is actually retrieved.

**Chunk metadata fields stored in ChromaDB:**

| Field | Description |
|---|---|
| `thread_id` | Tenant isolation key |
| `filename` | Source document name |
| `page` | 1-indexed page number |
| `chunk_type` | `text`, `table`, or `image` |
| `image_path` | Disk path for lazy VLM |
| `image_hash` | SHA-256 for cache lookup |
| `needs_vlm` | `True` if Gemini VLM hasn't run yet |
| `vlm_done` | `True` once Gemini VLM description is cached |

---

### B. Retrieval Pipeline (Multi-Stage Hybrid RAG)

```
[User Query]
      │
      ▼
┌─────────────────────────────────┐
│ 1. Query Expansion (Gemini)     │ → 3 alternative search variants
└─────────────────────────────────┘
      │ 4-query list
      ├──────────────────────────────────────┐
      ▼                                      ▼
┌──────────────────┐              ┌──────────────────┐
│ 2. Dense Search  │              │ 3. Sparse Search  │
│   (ChromaDB)     │              │   (BM25Okapi)     │
│   k=15/query     │              │   cached per      │
└──────────────────┘              │   thread (LRU 50) │
      │                           └──────────────────┘
      └──────────────┬────────────────────┘
                     ▼
        ┌───────────────────────────┐
        │ 4. Reciprocal Rank Fusion │ k=60, merges dense + sparse
        └───────────────────────────┘
                     │ Top-30 candidates
                     ▼
        ┌───────────────────────────┐
        │ 5. CrossEncoder Reranking │ BAAI/bge-reranker-base → Top-6
        └───────────────────────────┘
                     │
                     ▼
        ┌───────────────────────────┐
        │ 6. Lazy VLM Resolver      │ For image chunks:
        │   (Gemini Flash)          │   Check SQLite cache → hit: use cached
        └───────────────────────────┘   miss: call Gemini, cache result
                     │
                     ▼
              [Formatted Context → LLM]
```

**RRF Formula:**
$$RRF\_Score(d) = \sum_{m} \frac{1}{60 + Rank_m(d)}$$

**BM25 tokenizer** strips 28 English stopwords and tokens of length ≤ 1. Indices are cached per thread up to 50 entries (LRU eviction logged at WARNING level).

---

### C. Agent Graph & Memory

The LangGraph `StateGraph` is defined in `app/agent/graph.py`.

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    language: str
    summary: str
    last_summarized_count: int
```

```
[START] → chat_node → (tools called?) → Yes → tools_node → back to chat_node
                                       → No  → [END]
```

**Memory management:**
- **Sliding trim**: Latest 10 messages kept in context per invocation (`safe_trim_messages`).
- **Auto-summarization**: If >15 new messages since last summary, a background thread runs Gemini to condense older history → injected into system prompt.
- **Dynamic system prompt**: Assembles file list from PostgreSQL `thread_files` + ChromaDB metadata per thread.

**LLM fallback chain (`invoke_llm_with_fallback`):**
- Primary: `gemini-3.1-flash-lite`
- Retries up to 3× on 429 / 503 errors with cancellation-aware sleep
- Switches to `gemini-2.5-flash` on second attempt
- All retry events logged at `WARNING` level

---

### D. Concurrency & Stream Control

| Mechanism | Purpose |
|---|---|
| `_bm25_lock` (threading.Lock) | Serializes BM25 index reads/writes |
| `_rapid_ocr_lock` (threading.Lock) | Singleton RapidOCR instance protection |
| `cancelled_threads` (set + Lock) | Cooperative cancellation registry |
| `current_thread_id` (ContextVar) | Tracks active thread during execution |
| `asyncio.Queue` | SSE producer → consumer bridge |
| `run_in_executor` | Offloads blocking LangGraph calls from async event loop |

---

## 6. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the main UI (`index.html`) |
| `/languages` | GET | Returns 40 Gemini-fetched supported languages |
| `/history/{thread_id}` | GET | Full message history with timestamps |
| `/chat` | POST | Synchronous chat (returns complete reply) |
| `/chat_stream` | POST | SSE streaming chat (token-by-token) |
| `/stop` | POST | Cancels active stream for a thread |
| `/upload-pdf` | POST | Ingests PDF or DOCX into RAG pipeline (50 MB limit) |
| `/upload-excel` | POST | Saves Excel/CSV to workspace for Python analysis (50 MB limit) |
| `/transcribe` | POST | Transcribes audio via Whisper (25 MB limit, auto-detects format) |
| `/api/files/{filename}` | DELETE | Removes file from ChromaDB, disk, and DB registry |
| `/threads` | GET | Lists all saved thread sessions with metadata |
| `/health` | GET | Returns tool list and config status |
| `/debug/tools` | GET | Detailed tool registry debug info |

**SSE Event Types** (`/chat_stream`):
- `{"type": "text", "content": "<token>"}` — streamed assistant token
- `{"type": "tool_start", "name": "<tool>", "args": {}}` — tool invocation start
- `{"type": "tool_end", "name": "<tool>", "output": "<json>"}` — tool result
- `{"type": "meta", "timestamp": "<ISO>"}` — final message timestamp

**Error responses**: All endpoints return `{"detail": "An internal error occurred."}` — raw tracebacks are never sent to clients. Full tracebacks are written to `logs/app.log` with `exc_info=True`.

---

## 7. MCP & Tool Integrations

### Filesystem MCP (`filesystem`)
- Transport: Stdio → `npx -y @modelcontextprotocol/server-filesystem <MCP_FS_ROOT>`
- Write tools are removed at startup: `filesystem_write_file`, `filesystem_edit_file`, `filesystem_create_directory`, `filesystem_move_file`
- File writes go through the secure `write_file` tool which validates paths against `is_path_sensitive()`

### Playwright MCP (`playwright`)
- Transport: Stdio → `npx -y @executeautomation/playwright-mcp-server`
- Browser smoke test runs at startup (`about:blank`); if it fails, Playwright tools are excluded from the agent
- All executions use headless mode

### Sync Bridge (`make_sync_run`)
Wraps async MCP tools for synchronous LangGraph execution:
- Schedules onto main event loop via `asyncio.run_coroutine_threadsafe` if loop is running
- Falls back to `asyncio.run` if no loop active
- Applies `sanitize_tool_output` and `is_path_sensitive` on all results

### Built-in Tools
| Tool | Description |
|---|---|
| `web_search` | DuckDuckGo search via `duckduckgo_search` |
| `calculator` | Safe math expression evaluator |
| `stock_price` | AlphaVantage live market data |
| `python_interpreter` | Sandboxed Python REPL (AST-checked before execution) |
| `search_knowledge_base` | Full hybrid RAG pipeline |
| `read_file` / `write_file` | Sandboxed filesystem access |
| `playwright_playwright_navigate` | Ephemeral Playwright RAG (PDF capture → index → query → cleanup) |

---

## 8. Security

### Path Sandboxing (`is_path_sensitive`)
Blocks access to:
- Hidden files/folders (`.env`, `.git`, anything starting with `.`)
- Python sources (`.py`, `.pyc`, `__pycache__`)
- System databases (`chroma_db/`, `image_cache.db`)
- Core app code (`app/`, `backend.py`)
- Frontend sources (`static/*.js`, `static/*.html`, `static/*.css`)

### Python REPL Code Safety (`is_code_safe`)
AST-parsed before execution. Bans:
- **Imports**: `os`, `sys`, `subprocess`, `shutil`, `socket`, `urllib`, `requests`, `importlib`, `ctypes`, `pty`, `builtins`
- **Functions**: `eval`, `exec`, `compile`, `open`, `getattr`, `setattr`, `delattr`, `remove`, `unlink`, `rmdir`, `system`, `popen`, `fork`
- **Literals**: strings containing `.env`, `backend.py`, `chroma_db`, `scratch/`, `app/`

### Output Sanitizer (`sanitize_tool_output`)
Strips lines referencing internal paths, database names, or source files from all tool outputs before they reach the agent or frontend.

### Upload Security
- All uploads read into memory first, size checked, then written to temp file
- Excel/CSV files saved to `MCP_FS_ROOT` (not `os.getcwd()`)
- Temp files always cleaned up in `finally` blocks

---

## 9. Development & Deployment

### Prerequisites
- Python 3.10+ with Conda environment (`llm.yml`)
- PostgreSQL instance
- Node.js + npm (for MCP servers)
- LibreOffice (for DOCX → PDF conversion)

### Database Setup
```sql
CREATE DATABASE chatbot_db;
CREATE USER harsh WITH PASSWORD 'harsh';
GRANT ALL PRIVILEGES ON DATABASE chatbot_db TO harsh;
```

Tables (`thread_metadata`, `thread_files`, `message_timestamps`) are auto-created on first startup.

### Environment File (`.env`)
```env
GOOGLE_API_KEY=your_gemini_api_key
DB_URI="postgresql://harsh:harsh@localhost:5432/chatbot_db"
MCP_FS_ROOT=/path/to/workspace
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=Langsmith-demo
```

### Running the Server
```bash
conda activate llm
python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000` to access the chat interface.

### Viewing Logs
```bash
# Live tail
tail -f logs/app.log

# Filter errors only
grep "ERROR" logs/app.log

# Filter by module
grep "app.services.rag_pipeline" logs/app.log
```

### Production Deployment Notes
- Replace `--reload` with `--workers 2` for production
- Move API keys from `.env` to a secret manager (AWS Secrets, Vault, etc.)
- Back up `chroma_db/` and `image_cache.db` regularly — these are not replicated
- The `logs/` directory is git-ignored; set up log shipping (e.g. Loki, ELK) for cloud deployments

---

## 10. Known Gaps & Future Improvements

| Item | Priority | Notes |
|---|---|---|
| No authentication/authorization | 🔴 High | Any client can access any thread. Add API key header or OAuth. |
| No rate limiting | 🔴 High | Add `slowapi` middleware to protect `/chat`, `/upload-pdf` |
| No Dockerfile / docker-compose | 🟠 Medium | Required for reproducible deployment |
| `asyncio.get_event_loop()` deprecation | 🟠 Medium | Replace with `get_running_loop()` throughout |
| ContextVar not propagated to threads | 🟠 Medium | Wrap producer with `copy_context().run(producer)` |
| Query expansion hardcoded for finance | 🟡 Low | Prompt says "financial document" — make it domain-agnostic |
| No ChromaDB backup strategy | 🟡 Low | Add scheduled backup of `chroma_db/` folder |
| No Prometheus metrics | 🟡 Low | Add request latency, RAG hit rate counters |

---

**LangGraph Chatbot v16 — Technical Reference Manual**  
*Production-hardened build with structured logging, RapidOCR ingestion, Gemini Flash VLM, and hybrid RAG retrieval.*
