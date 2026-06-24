import os
import json
import asyncio
import traceback
import tempfile
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import psycopg
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from datetime import datetime, timezone
from langchain_mcp_adapters.client import MultiServerMCPClient

# Modular imports
from app.core.config import DB_URI, MCP_SERVERS, MCP_FS_ROOT
from app.core.logger import setup_logging, get_logger
from app.core.db import get_db_connection, get_checkpointer

logger = get_logger(__name__)
from app.core.security import sanitize_value
from app.agent.tools import (
    web_search, calculator, stock_price, write_file, read_file, 
    list_directory, python_interpreter, make_sync_run, index_local_file,
    playwright_playwright_navigate
)
from app.services.rag_pipeline import (
    search_knowledge_base, index_pdf_file, index_docx_file, get_embeddings,
    get_vectorstore, get_cross_encoder, invalidate_bm25_cache, sanitize_filename
)
from app.services.audio import get_whisper_model, transcribe_audio_file
from app.agent.graph import create_chatbot, ChatState, cancel_thread, is_thread_cancelled, clear_thread_cancellation, current_thread_id

# Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str = Field(default="default")
    language: str = Field(default="English")


class StopRequest(BaseModel):
    thread_id: str


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    timestamp: str = None



class ToolCallOut(BaseModel):
    name: str
    args: dict
    output: str = None


class MessageOut(BaseModel):
    role: str   # "user" or "assistant"
    content: str
    timestamp: str = None
    tool_calls: list[ToolCallOut] = None


# Globals
chatbot = None
_checkpointer = None
_mcp_client = None
_loaded_tool_names: list[str] = []
_main_loop = None
_supported_languages = []
_languages_lock = asyncio.Lock()

def get_main_loop():
    return _main_loop



@asynccontextmanager
async def lifespan(_app: FastAPI):
    global chatbot, _checkpointer, _mcp_client, _loaded_tool_names, _main_loop
    _main_loop = asyncio.get_running_loop()
    setup_logging()

    # Preload models to prevent delays on first queries
    try:
        get_cross_encoder()
    except Exception as e:
        logger.warning(f"Failed to pre-load CrossEncoder: {e}")

    try:
        get_whisper_model()
    except Exception as e:
        logger.warning(f"Failed to pre-load Whisper model: {e}")

    try:
        get_embeddings()
    except Exception as e:
        logger.warning(f"Failed to pre-load HuggingFaceEmbeddings: {e}")

    try:
        get_vectorstore()
    except Exception as e:
        logger.warning(f"Failed to pre-load Chroma vectorstore: {e}")

    builtin_tools = [
        web_search, calculator, stock_price,
        write_file, read_file, list_directory,
        search_knowledge_base, python_interpreter, index_local_file,
        playwright_playwright_navigate,
    ]

    # MCP tools — use get_tools() which creates ephemeral sessions per tool call
    # This avoids cancel scope / async generator lifecycle errors
    mcp_tools = []

    for name, srv_config in MCP_SERVERS.items():
        try:
            client = MultiServerMCPClient({name: srv_config}, tool_name_prefix=True)
            tools = await client.get_tools()

            # Restrict filesystem tools to read-only
            if name == "filesystem":
                dangerous_tools = {
                    "filesystem_write_file",
                    "filesystem_edit_file",
                    "filesystem_create_directory",
                    "filesystem_move_file"
                }
                tools = [t for t in tools if t.name not in dangerous_tools]
                logger.info(f"[MCP] Filesystem: removed write/modify tools. Remaining: {[t.name for t in tools]}")

            for t in tools:
                t._run = make_sync_run(t, get_main_loop)

            # Verify browser functionality if playwright
            if name == "playwright":
                try:
                    navigate_tool = next((t for t in tools if "navigate" in t.name), None)
                    if navigate_tool:
                        logger.info("[MCP] Playwright: sleeping 2s for server setup...")
                        await asyncio.sleep(2.0)
                        logger.info("[MCP] Playwright: verifying browser...")
                        await asyncio.wait_for(navigate_tool.ainvoke({"url": "about:blank", "headless": True}), timeout=25.0)
                        logger.info("[MCP] Playwright: browser check OK.")
                    else:
                        logger.warning("[MCP] Playwright: no navigate tool found.")
                except Exception as p_err:
                    logger.error(f"[MCP] Playwright check FAILED: {p_err} — skipping.", exc_info=True)
                    continue

            mcp_tools.extend(tools)
            logger.info(f"[MCP] ✓ '{name}' — {len(tools)} tool(s): {[t.name for t in tools]}")
            if _mcp_client is None:
                _mcp_client = client
        except Exception:
            logger.error(f"[MCP] ✗ '{name}' failed — skipping.", exc_info=True)

    # Combine and ensure all tool names are unique
    all_tools = []
    seen_tool_names = set()
    for t in (builtin_tools + mcp_tools):
        if t.name not in seen_tool_names:
            seen_tool_names.add(t.name)
            all_tools.append(t)
            
    _loaded_tool_names = [t.name for t in all_tools]
    logger.info(f"[App] Tools available ({len(_loaded_tool_names)}): {_loaded_tool_names}")

    logger.info(f"[App] Connecting to DB...")
    checkpointer_class, checkpointer_cm = get_checkpointer()
    _checkpointer = checkpointer_cm.__enter__()
    if hasattr(_checkpointer, "setup"):
        _checkpointer.setup()

    # Initialize thread metadata and files tables
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS thread_metadata (
                        thread_id TEXT PRIMARY KEY,
                        thread_name TEXT NOT NULL
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS thread_files (
                        thread_id TEXT,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        PRIMARY KEY (thread_id, filename)
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS message_timestamps (
                        thread_id TEXT,
                        message_id TEXT,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (thread_id, message_id)
                    );
                """)
                conn.commit()
        logger.info("[App] DB tables initialised successfully.")
    except Exception as e:
        logger.error(f"[App] Error initialising DB tables: {e}", exc_info=True)

    chatbot = create_chatbot(all_tools, _checkpointer)
    logger.info("[App] Ready.")

    yield

    try:
        checkpointer_cm.__exit__(None, None, None)
    except BaseException:
        pass
    logger.info("[App] Shutdown complete.")


app = FastAPI(title="LangGraph Chatbot", lifespan=lifespan)

# Serve static files
# Make sure we point correctly to static/ directory
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


async def get_supported_languages_cached():
    global _supported_languages
    if _supported_languages:
        return _supported_languages
        
    async with _languages_lock:
        if _supported_languages:
            return _supported_languages
            
        try:
            logger.info("[App] Fetching supported languages from Gemini...")
            from langchain_google_genai import ChatGoogleGenerativeAI
            temp_model = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite",
                thinking_budget=0,
                temperature=0.0,
            )
            lang_prompt = (
                "You are a system utility. Return a JSON list of major languages you officially support and can converse in. "
                "Format the response as a JSON array of objects, where each object has 'code' (ISO 639-1 language code, e.g., 'en', 'es', 'fr', 'hi') and 'name' (the English name of the language, e.g., 'English', 'Spanish', 'French', 'Hindi'). "
                "Include the top 35-40 most popular/common languages globally. "
                "Do not include any markdown styling, quotes, or conversational text. Return ONLY the raw JSON string starting with [ and ending with ]."
            )
            res = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: temp_model.invoke(lang_prompt)
            )
            content = res.content
            if isinstance(content, list):
                content = " ".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
            elif not isinstance(content, str):
                content = str(content)
            
            content_clean = content.strip()
            if content_clean.startswith("```json"):
                content_clean = content_clean.replace("```json", "", 1)
            if content_clean.endswith("```"):
                content_clean = content_clean.rsplit("```", 1)[0]
            content_clean = content_clean.strip()
            
            _supported_languages = json.loads(content_clean)
            logger.info(f"[App] Loaded {len(_supported_languages)} supported languages dynamically.")
        except Exception as lang_err:
            logger.warning(f"[App] Failed to fetch supported languages dynamically: {lang_err} — using fallback list.")
            _supported_languages = [
                {"code": "en", "name": "English"},
                {"code": "es", "name": "Spanish"},
                {"code": "fr", "name": "French"},
                {"code": "de", "name": "German"},
                {"code": "zh", "name": "Chinese"},
                {"code": "ja", "name": "Japanese"},
                {"code": "hi", "name": "Hindi"},
                {"code": "pt", "name": "Portuguese"},
                {"code": "ru", "name": "Russian"},
                {"code": "it", "name": "Italian"},
                {"code": "ar", "name": "Arabic"},
                {"code": "ko", "name": "Korean"},
                {"code": "tr", "name": "Turkish"},
                {"code": "vi", "name": "Vietnamese"},
                {"code": "nl", "name": "Dutch"}
            ]
        return _supported_languages


@app.get("/languages")
async def get_languages():
    langs = await get_supported_languages_cached()
    return {"languages": langs}



def generate_and_save_title(thread_id: str, message_text: str):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT thread_name FROM thread_metadata WHERE thread_id = %s", (thread_id,))
                row = cur.fetchone()
                if row:
                    return
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not os.getenv("GOOGLE_API_KEY"):
            return
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            thinking_budget=0,
            temperature=0.7,
        )
        prompt = (
            "You are a utility assistant. Summarize the following user's first query into a short, concise chat title of 3 to 5 words. "
            "Do not include quotes, prefixes like 'Title:', or markdown. Reply with ONLY the title.\n\n"
            f"User Query: {message_text}"
        )
        res = llm.invoke(prompt)
        content = res.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        elif not isinstance(content, str):
            content = str(content)
            
        title = content.strip().strip('"').strip("'")
        if not title:
            title = f"Chat {thread_id[:8]}"
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO thread_metadata (thread_id, thread_name)
                    VALUES (%s, %s)
                    ON CONFLICT (thread_id)
                    DO UPDATE SET thread_name = EXCLUDED.thread_name;
                    """,
                    (thread_id, title)
                )
                conn.commit()
        logger.info(f"[App] Auto-generated title for thread {thread_id}: '{title}'")
    except Exception as e:
        logger.error(f"[App] Error auto-generating thread title: {e}", exc_info=True)


@app.get("/health")
async def health():
    return {"status": "ok", "tools_loaded": _loaded_tool_names}


@app.get("/debug/tools")
async def debug_tools():
    return {
        "total": len(_loaded_tool_names),
        "tools": _loaded_tool_names,
        "fs_root": MCP_FS_ROOT,
    }


class RenameRequest(BaseModel):
    thread_name: str


@app.post("/threads/{thread_id}/rename")
async def rename_thread(thread_id: str, request: RenameRequest):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO thread_metadata (thread_id, thread_name)
                    VALUES (%s, %s)
                    ON CONFLICT (thread_id)
                    DO UPDATE SET thread_name = EXCLUDED.thread_name;
                    """,
                    (thread_id, request.thread_name)
                )
                conn.commit()
        return {"status": "success", "thread_id": thread_id, "thread_name": request.thread_name}
    except Exception as e:
        logger.error(f"[App] Error renaming thread {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM thread_metadata WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM thread_files WHERE thread_id = %s", (thread_id,))
                conn.commit()
        return {"status": "success", "thread_id": thread_id}
    except Exception as e:
        logger.error(f"[App] Error deleting thread {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}/documents")
async def get_thread_documents(thread_id: str):
    try:
        seen_filenames = set()
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT filename FROM thread_files WHERE thread_id = %s",
                        (thread_id,)
                    )
                    rows = cur.fetchall()
                    for (fname,) in rows:
                        seen_filenames.add(fname)
        except Exception as db_err:
            logger.error(f"[App] DB query for thread documents failed: {db_err}", exc_info=True)

        # Get count of chunks for each filename in Chroma
        vectorstore = get_vectorstore()
        counts = {}
        try:
            all_data = vectorstore.get(where={"thread_id": thread_id}, include=["metadatas"])
            if all_data and "metadatas" in all_data and all_data["metadatas"]:
                for meta in all_data["metadatas"]:
                    if meta and "filename" in meta:
                        fn = meta["filename"]
                        counts[fn] = counts.get(fn, 0) + 1
                    elif meta and "source" in meta:
                        src = meta["source"]
                        if src:
                            fn = os.path.basename(src)
                            counts[fn] = counts.get(fn, 0) + 1
        except Exception as err:
            logger.warning(f"[App] Error getting chunk counts from Chroma: {err}")

        if not seen_filenames:
            for fn in counts:
                seen_filenames.add(fn)

        documents_info = []
        for fname in sorted(list(seen_filenames)):
            documents_info.append({"filename": fname, "chunks": counts.get(fname, 0)})

        return {"documents": documents_info}
    except Exception as e:
        logger.error(f"[App] Error listing thread documents: {e}", exc_info=True)
        return {"documents": []}


@app.get("/threads")
async def list_threads():
    if _checkpointer is None:
        return {"threads": []}
    seen = set()
    thread_ids = []
    try:
        for checkpoint in _checkpointer.list(None):
            tid = (
                checkpoint.config
                .get("configurable", {})
                .get("thread_id")
            )
            if tid and tid not in seen:
                seen.add(tid)
                thread_ids.append(tid)
    except Exception:
        pass

    thread_names = {}
    if thread_ids:
        try:
            placeholders = ",".join(["%s"] * len(thread_ids))
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT thread_id, thread_name FROM thread_metadata WHERE thread_id IN ({placeholders})",
                        tuple(thread_ids)
                    )
                    rows = cur.fetchall()
                    for tid_db, name_db in rows:
                        thread_names[tid_db] = name_db
        except Exception as e:
            logger.error(f"[App] Error fetching thread names: {e}", exc_info=True)

    threads_with_metadata = []
    for tid in thread_ids:
        display_name = thread_names.get(tid, f"{tid[:8]}…")
        threads_with_metadata.append({"thread_id": tid, "thread_name": display_name})

    return {"threads": threads_with_metadata}


def save_message_timestamps(thread_id: str, messages: list):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for msg in messages:
                    msg_id = getattr(msg, "id", None)
                    if msg_id:
                        # Extract timestamp from message object additional_kwargs if available
                        timestamp = None
                        if hasattr(msg, "additional_kwargs") and isinstance(msg.additional_kwargs, dict):
                            timestamp = msg.additional_kwargs.get("timestamp")
                        
                        if timestamp:
                            cur.execute(
                                """
                                INSERT INTO message_timestamps (thread_id, message_id, timestamp)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (thread_id, message_id) DO NOTHING
                                """,
                                (thread_id, msg_id, timestamp)
                            )
                        else:
                            cur.execute(
                                """
                                INSERT INTO message_timestamps (thread_id, message_id, timestamp)
                                VALUES (%s, %s, NOW())
                                ON CONFLICT (thread_id, message_id) DO NOTHING
                                """,
                                (thread_id, msg_id)
                            )
                conn.commit()
    except Exception as e:
        logger.error(f"[App] Error saving message timestamps for {thread_id}: {e}", exc_info=True)


def get_message_timestamps(thread_id: str) -> dict[str, str]:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT message_id, timestamp FROM message_timestamps WHERE thread_id = %s",
                    (thread_id,)
                )
                rows = cur.fetchall()
                return {row[0]: row[1].isoformat() if hasattr(row[1], "isoformat") else str(row[1]) for row in rows}
    except Exception as e:
        logger.error(f"[App] Error getting message timestamps for {thread_id}: {e}", exc_info=True)
        return {}


@app.get("/history/{thread_id}", response_model=list[MessageOut])
async def get_history(thread_id: str):
    if chatbot is None:
        return []
    state = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: chatbot.get_state({"configurable": {"thread_id": thread_id}}),
    )
    if not state:
        return []
    messages = state.values.get("messages", [])
    
    # Save missing timestamps
    if messages:
        await asyncio.get_event_loop().run_in_executor(
            None,
            save_message_timestamps,
            thread_id,
            messages
        )
        
    timestamps = await asyncio.get_event_loop().run_in_executor(
        None,
        get_message_timestamps,
        thread_id
    )
    
    # Map tool_call_id to ToolMessage content
    tool_outputs = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_outputs[msg.tool_call_id] = str(msg.content)

    result = []
    current_tool_calls = []

    for msg in messages:
        msg_id = getattr(msg, "id", None)
        # Check additional_kwargs first, fallback to DB timestamps
        timestamp_str = None
        if hasattr(msg, "additional_kwargs") and isinstance(msg.additional_kwargs, dict):
            timestamp_str = msg.additional_kwargs.get("timestamp")
        if not timestamp_str and msg_id:
            timestamp_str = timestamps.get(msg_id)
        
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            result.append(MessageOut(role="user", content=content, timestamp=timestamp_str, tool_calls=[]))
            # Reset any accumulated tool calls for the next assistant message
            current_tool_calls = []
        elif isinstance(msg, AIMessage):
            has_tc = False
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                has_tc = True
                for tc in msg.tool_calls:
                    tc_id = tc.get("id")
                    current_tool_calls.append(ToolCallOut(
                        name=tc.get("name"),
                        args=tc.get("args") or {},
                        output=tool_outputs.get(tc_id, "No output recorded.")
                    ))
            
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            elif not isinstance(content, str):
                content = str(content)
            
            if content.strip() or (has_tc and msg == messages[-1]):
                result.append(MessageOut(
                    role="assistant",
                    content=content,
                    timestamp=timestamp_str,
                    tool_calls=list(current_tool_calls)
                ))
                current_tool_calls = []
                
    return result


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialised yet.")

    try:
        background_tasks.add_task(generate_and_save_title, request.thread_id, request.message)

        config: RunnableConfig = {"configurable": {"thread_id": request.thread_id}}

        human_msg = HumanMessage(
            content=request.message,
            additional_kwargs={"timestamp": datetime.now(timezone.utc).isoformat()}
        )
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: chatbot.invoke(
                {
                    "messages": [human_msg],
                    "language": request.language
                },
                config,
            ),
        )

        # Save timestamps for newly added messages
        await asyncio.get_event_loop().run_in_executor(
            None,
            save_message_timestamps,
            request.thread_id,
            result["messages"][-2:]
        )

        reply = result["messages"][-1]
        if isinstance(reply, AIMessage):
            content = reply.content
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            elif not isinstance(content, str):
                content = str(content)
        else:
            content = str(reply)

        # Fetch newly saved reply timestamp
        reply_id = getattr(reply, "id", None)
        timestamp_str = None
        if reply_id:
            timestamps = await asyncio.get_event_loop().run_in_executor(
                None,
                get_message_timestamps,
                request.thread_id
            )
            timestamp_str = timestamps.get(reply_id)

        return ChatResponse(reply=content, thread_id=request.thread_id, timestamp=timestamp_str)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[App] /chat error for thread {request.thread_id}:\n{tb}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


active_tasks = {}

@app.post("/stop")
async def stop_chat(request: StopRequest):
    thread_id = request.thread_id
    cancel_thread(thread_id)
    if thread_id in active_tasks:
        task = active_tasks[thread_id]
        task.cancel()
        logger.info(f"[App] Cancelled active stream task for thread {thread_id}.")
        return {"status": "success", "message": f"Cancelled active response for thread {thread_id}"}
    return {"status": "success", "message": f"No active response found for thread {thread_id}"}


@app.post("/chat_stream")
async def chat_stream(chat_req: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialised yet.")

    background_tasks.add_task(generate_and_save_title, chat_req.thread_id, chat_req.message)

    config: RunnableConfig = {"configurable": {"thread_id": chat_req.thread_id}}

    clear_thread_cancellation(chat_req.thread_id)

    if chat_req.thread_id in active_tasks:
        try:
            active_tasks[chat_req.thread_id].cancel()
            logger.info(f"[App] Cancelled existing task for thread {chat_req.thread_id} before new stream.")
        except Exception as cancel_err:
            logger.warning(f"[App] Error cancelling existing task: {cancel_err}")

    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def producer():
        try:
            current_thread_id.set(chat_req.thread_id)
            human_msg = HumanMessage(
                content=chat_req.message,
                additional_kwargs={"timestamp": datetime.now(timezone.utc).isoformat()}
            )
            for msg, metadata in chatbot.stream(
                {
                    "messages": [human_msg],
                    "language": chat_req.language
                },
                config,
                stream_mode="messages"
            ):
                if is_thread_cancelled(chat_req.thread_id):
                    logger.info(f"[App] Producer for thread {chat_req.thread_id}: cancellation detected, aborting stream.")
                    break
                loop.call_soon_threadsafe(queue.put_nowait, (msg, metadata))
        except Exception as e:
            logger.error(f"[App] Stream producer error for thread {chat_req.thread_id}: {e}", exc_info=True)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, (None, None))

    task = loop.run_in_executor(None, producer)
    active_tasks[chat_req.thread_id] = task

    async def event_generator():
        seen_tool_calls = set()
        try:
            while True:
                if await request.is_disconnected():
                    logger.info(f"[App] Client disconnected for thread {chat_req.thread_id}.")
                    cancel_thread(chat_req.thread_id)
                    task.cancel()
                    break

                if is_thread_cancelled(chat_req.thread_id):
                    logger.info(f"[App] event_generator for thread {chat_req.thread_id} cancelled.")
                    task.cancel()
                    break

                try:
                    msg, metadata = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if msg is None:
                    break

                if isinstance(msg, AIMessage):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tc_id = tc.get("id")
                            if tc_id and tc_id not in seen_tool_calls:
                                seen_tool_calls.add(tc_id)
                                yield json.dumps({
                                    "type": "tool_start",
                                    "name": tc.get("name"),
                                    "args": tc.get("args")
                                }) + "\n"

                    content = msg.content
                    if isinstance(content, list):
                        content = " ".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    elif not isinstance(content, str):
                        content = str(content)

                    if content:
                        yield json.dumps({
                            "type": "text",
                            "content": content
                        }) + "\n"

                elif isinstance(msg, ToolMessage):
                    yield json.dumps({
                        "type": "tool_end",
                        "name": getattr(msg, "name", "tool"),
                        "output": str(msg.content)
                    }) + "\n"

            if not is_thread_cancelled(chat_req.thread_id):
                try:
                    state = await loop.run_in_executor(
                        None,
                        lambda: chatbot.get_state(config)
                    )
                    if state:
                        msgs = state.values.get("messages", [])
                        if msgs:
                            await loop.run_in_executor(
                                None,
                                save_message_timestamps,
                                chat_req.thread_id,
                                msgs[-2:]
                            )
                            last_msg = msgs[-1]
                            last_msg_id = getattr(last_msg, "id", None)
                            if last_msg_id:
                                timestamps = await loop.run_in_executor(
                                    None,
                                    get_message_timestamps,
                                    chat_req.thread_id
                                )
                                timestamp_str = timestamps.get(last_msg_id)
                                if timestamp_str:
                                    yield json.dumps({
                                        "type": "meta",
                                        "timestamp": timestamp_str
                                    }) + "\n"
                except Exception as state_err:
                    logger.error(f"[App] Error saving stream timestamps: {state_err}", exc_info=True)

        except asyncio.CancelledError:
            logger.info(f"[App] event_generator cancelled for thread {chat_req.thread_id}.")
            cancel_thread(chat_req.thread_id)
            task.cancel()
        finally:
            if active_tasks.get(chat_req.thread_id) == task:
                active_tasks.pop(chat_req.thread_id, None)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), thread_id: str = "default"):
    MAX_SIZE = 50 * 1024 * 1024  # 50 MB
    try:
        filename = file.filename
        _, ext = os.path.splitext(filename.lower())

        file_bytes = await file.read()
        if len(file_bytes) > MAX_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 50 MB.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        try:
            if ext == ".docx":
                chunks_added = index_docx_file(temp_path, filename, thread_id, db_uri=DB_URI)
            else:
                chunks_added = index_pdf_file(temp_path, filename, thread_id, db_uri=DB_URI)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return {
            "status": "success",
            "message": f"Processed '{filename}' — added {chunks_added} chunks.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[App] PDF upload failed.", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during file processing.")


@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...), thread_id: str = "default"):
    MAX_SIZE = 50 * 1024 * 1024  # 50 MB
    try:
        filename = file.filename
        if not filename.lower().endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(status_code=400, detail="Only Excel (.xlsx, .xls) and CSV (.csv) files are supported.")

        file_bytes = await file.read()
        if len(file_bytes) > MAX_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 50 MB.")

        target_path = os.path.join(MCP_FS_ROOT, filename)
        with open(target_path, "wb") as buffer:
            buffer.write(file_bytes)

        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO thread_files (thread_id, filename, file_type)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (thread_id, filename) DO NOTHING;
                    """, (thread_id, filename, 'excel'))
                    conn.commit()
        except Exception as db_err:
            logger.error(f"[App] DB save for Excel file failed: {db_err}", exc_info=True)

        return {
            "status": "success",
            "message": f"Saved '{filename}' to workspace for data analysis.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[App] Excel/CSV upload failed.", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while saving the file.")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    MAX_SIZE = 25 * 1024 * 1024  # 25 MB
    try:
        _, ext = os.path.splitext((file.filename or "").lower())
        audio_ext = ext if ext in (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm") else ".wav"

        file_bytes = await file.read()
        if len(file_bytes) > MAX_SIZE:
            raise HTTPException(status_code=413, detail="Audio file too large. Maximum allowed size is 25 MB.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_ext) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        try:
            transcription = transcribe_audio_file(temp_path)
        except Exception as trans_err:
            logger.warning(f"[App] Transcription internal error: {trans_err}")
            raise trans_err
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return {"text": transcription}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[App] Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during transcription.")


@app.delete("/api/files/{filename}")
async def delete_file(filename: str, thread_id: str):
    try:
        # Check if the filename exists in thread_files for that thread
        exists = False
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM thread_files WHERE thread_id = %s AND filename = %s",
                        (thread_id, filename)
                    )
                    exists = cur.fetchone() is not None
        except Exception as db_err:
            logger.error(f"[App] DB query for file check failed: {db_err}", exc_info=True)
        
        if not exists:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found for thread '{thread_id}'")

        # 1. Delete ChromaDB chunks
        vectorstore = get_vectorstore()
        vectorstore.delete(where={
            "$and": [
                {"thread_id": {"$eq": thread_id}},
                {"filename": {"$eq": filename}}
            ]
        })

        # 2. Delete saved images from disk
        sanitized_name = sanitize_filename(filename)
        images_dir = os.path.join("static", "extracted_images", sanitized_name)
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir, ignore_errors=True)

        # Delete physical file from the workspace if it exists
        workspace_file_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(workspace_file_path):
            os.remove(workspace_file_path)

        # 3. Delete from thread_files table
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM thread_files WHERE thread_id = %s AND filename = %s",
                        (thread_id, filename)
                    )
                    conn.commit()
        except Exception as db_err:
            logger.error(f"[App] DB delete for file failed: {db_err}", exc_info=True)

        # 4. Invalidate BM25 cache
        invalidate_bm25_cache(thread_id)

        return {"success": True, "message": f"{filename} removed successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("[App] File deletion failed.", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
