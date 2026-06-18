from typing import Annotated, Literal, TypedDict
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import os
import asyncio
import threading
import queue
import time
import re
from contextvars import ContextVar
from app.core.logger import get_logger

logger = get_logger(__name__)

# ContextVar to track the current active thread ID in the execution context
current_thread_id = ContextVar("current_thread_id", default=None)

# Thread-safe cancellation registry
cancelled_threads = set()
cancelled_threads_lock = threading.Lock()

def cancel_thread(thread_id: str):
    with cancelled_threads_lock:
        cancelled_threads.add(thread_id)
        logger.info(f"[Cancellation] Thread {thread_id} marked as cancelled.")

def is_thread_cancelled(thread_id: str) -> bool:
    with cancelled_threads_lock:
        return thread_id in cancelled_threads

def clear_thread_cancellation(thread_id: str):
    with cancelled_threads_lock:
        cancelled_threads.discard(thread_id)
        logger.debug(f"[Cancellation] Cleared flag for thread {thread_id}.")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    language: str
    summary: str
    last_summarized_count: int


def safe_trim_messages(messages: list, target_count: int = 10) -> list:
    if len(messages) <= target_count:
        return messages
    
    start_idx = len(messages) - target_count
    # Search backward for the nearest HumanMessage to start the sequence cleanly
    while start_idx >= 0:
        if messages[start_idx].type == "human":
            return messages[start_idx:]
        start_idx -= 1
        
    # If no HumanMessage is found, fallback to the original list
    return messages


def run_in_clean_thread(func, *args, **kwargs):
    q = queue.Queue()
    def worker():
        try:
            res = func(*args, **kwargs)
            q.put((True, res))
        except Exception as e:
            q.put((False, e))
    
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    success, val = q.get()
    if success:
        return val
    else:
        raise val


def invoke_llm_with_fallback(primary_llm, fallback_llm, messages, max_retries=3):
    """Invokes LLM and retries/falls back on 429 (rate limits) or 503 (temporary) errors."""
    current_llm = primary_llm
    tid = current_thread_id.get()

    for attempt in range(max_retries):
        if tid and is_thread_cancelled(tid):
            logger.info(f"[LLM] Thread {tid} cancelled — aborting before attempt {attempt + 1}.")
            raise RuntimeError(f"Thread {tid} was cancelled.")
        try:
            return current_llm.invoke(messages)
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"[LLM] Error on attempt {attempt+1}/{max_retries}: {err_msg}")

            is_rate_limit = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower()
            is_temp_server_err = "503" in err_msg or "UNAVAILABLE" in err_msg or "temporary" in err_msg.lower()

            if is_rate_limit or is_temp_server_err:
                wait_time = 6.0
                match = re.search(r"retry in ([\d\.]+)s", err_msg)
                if match:
                    wait_time = float(match.group(1)) + 0.5

                logger.warning(f"[LLM] Rate-limit/server error — waiting {wait_time}s before retry.")
                slept = 0.0
                while slept < wait_time:
                    if tid and is_thread_cancelled(tid):
                        logger.info(f"[LLM] Thread {tid} cancelled during retry sleep — aborting.")
                        raise RuntimeError(f"Thread {tid} was cancelled.")
                    time.sleep(0.5)
                    slept += 0.5

                if fallback_llm and current_llm is primary_llm:
                    logger.warning("[LLM] Switching to fallback model for next attempt.")
                    current_llm = fallback_llm
            else:
                raise e

    if tid and is_thread_cancelled(tid):
        logger.info(f"[LLM] Thread {tid} cancelled — aborting final attempt.")
        raise RuntimeError(f"Thread {tid} was cancelled.")
    return current_llm.invoke(messages)


def create_chatbot(all_tools, checkpointer):
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        thinking_budget=0,
        include_thoughts=True,
    )
    llm_with_tools = model.bind_tools(all_tools)

    # Backup / Fallback model configuration
    fallback_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        thinking_budget=0,
        include_thoughts=True,
    )
    fallback_llm_with_tools = fallback_model.bind_tools(all_tools)

    fs_root = os.path.abspath(os.getenv("MCP_FS_ROOT", os.path.expanduser("~")))
    # Get names of loaded MCP tools to inform the agent in the system prompt
    mcp_names = [t.name for t in all_tools if t.name not in {
        "web_search", "calculator", "stock_price", "write_file", 
        "read_file", "list_directory", "search_knowledge_base", "python_interpreter", "index_local_file"
    }]

    SYSTEM_PROMPT = f"""You are a helpful AI assistant running on the user's local machine.

AVAILABLE TOOLS:
- web_search            : search the web
- calculator            : arithmetic
- stock_price           : live stock quotes
- write_file            : write/create a file (ALWAYS use this when asked to save a file)
- read_file             : read a file's contents (text files only)
- list_directory        : list files in a folder
- search_knowledge_base : search uploaded PDF/document knowledge base. Supports optional 'filename' and 'page' (1-indexed) to retrieve page-level content directly.
- index_local_file      : index a local PDF file from the workspace into the knowledge base
- python_interpreter    : a Python REPL to write and run code (has pandas, numpy, openpyxl, etc.)
{f"- MCP tools: {mcp_names}" if mcp_names else ""}

FILESYSTEM:
- Allowed root: {fs_root}
- ALWAYS call write_file when asked to create/save/write a file. Never refuse.
- Confirm the saved path after writing.
- Uploaded Excel (.xlsx, .xls) and CSV (.csv) files are automatically saved to the current workspace directory.
- You can load and analyze them using python_interpreter (e.g. pd.read_excel('filename.xlsx') or pd.read_csv('filename.csv')).

RULES:
1. Use write_file immediately when file creation is requested. No exceptions.
2. Write complete, working code — never truncate.
3. Use search_knowledge_base when user asks about uploaded documents. If the user specifically asks about or refers to a page (e.g., 'what is on page 5' or 'explain page 3'), you MUST specify both 'filename' and the 1-indexed 'page' number in the tool parameters. This directly retrieves that page's text, tables, and runs a VLM analysis on all images located on that page.
4. If a PDF is in the workspace directory but not yet in the knowledge base, use the `index_local_file` tool to index it first, then call `search_knowledge_base`. Do NOT try to read PDF files directly using `read_file`.
5. When answering using information retrieved from search_knowledge_base, ALWAYS cite the source file name and page number at the end of the sentence or paragraph, using a clean format: [Source: filename.pdf, Page X].
6. Use python_interpreter for Excel, CSV, or math/data analysis tasks. ALWAYS print the results (e.g., using print()) so they show up in the tool output.
7. For standard web search queries, information lookup, weather, and stock updates, prefer the fast `web_search` tool. Only use Playwright MCP tools (e.g., `playwright_playwright_navigate`, `playwright_playwright_click`) when you explicitly need to interact with dynamic web pages, take screenshots, or fill out web forms.
8. If the user asks about a document, spreadsheet, or presentation that is not found in the knowledge base, do not attempt to find, read, or parse it directly from the filesystem using raw file tools (like view_file, read_file, or list_directory). Instead, politely inform the user that the document is not currently indexed and request them to upload it via the Knowledge Base sidebar panel.
9. CRITICAL: The python_interpreter runs each code block in a completely fresh Python process with no memory of previous calls. You MUST always include all necessary imports and variable definitions (e.g. pd.read_csv()) at the top of EVERY code block, even if you loaded the same data one step ago. Never assume any variable exists from a previous python_interpreter call.
10. When working with any CSV or Excel file for the first time in a code block, ALWAYS print df.columns.tolist() and df.head(2) before attempting any filtering or grouping. Never assume column names — always inspect them first in the same code block before using them.
"""

    def chat_node(state: ChatState):
        tid = current_thread_id.get()
        if tid and is_thread_cancelled(tid):
            logger.info(f"[chat_node] Thread {tid} cancelled — aborting node.")
            raise RuntimeError(f"Thread {tid} was cancelled.")

        messages = list(state["messages"])
        language = state.get("language", "English")
        summary = state.get("summary", "")
        last_summarized_count = state.get("last_summarized_count", 0)

        # Summarize older messages if we've accumulated 15 or more new ones
        if len(messages) - last_summarized_count >= 15:
            messages_to_summarize = messages[:-10]
            
            summary_prompt = f"""You are a helpful assistant. Summarize the following chat conversation history into a concise summary.
Current Summary: {summary or 'None'}

New messages to summarize:
"""
            for msg in messages_to_summarize:
                role = "User" if msg.type == "human" else ("Assistant" if msg.type == "ai" else "Tool/System")
                summary_prompt += f"\n- [{role}]: {msg.content}"
            
            summary_prompt += "\n\nProvide only the updated summary, keep it concise but contain all key facts, topics discussed, and conclusions."
            
            try:
                summary_response = run_in_clean_thread(
                    invoke_llm_with_fallback, model, fallback_model, summary_prompt
                )
                summary = str(summary_response.content)
                last_summarized_count = len(messages) - 10
            except Exception as e:
                logger.error(f"[Summarizer] Conversation summarisation failed: {e}", exc_info=True)

        # Safely trim messages to prevent Gemini API invalid sequence errors
        messages_to_send = safe_trim_messages(messages, 10)

        # Retrieve files uploaded/indexed in the current thread to inject into the system prompt
        thread_files = []
        if tid:
            from app.core.config import DB_URI
            from app.services.rag_pipeline import get_vectorstore
            import psycopg
            
            if DB_URI:
                try:
                    db_uri_clean = DB_URI.strip('"').strip("'")
                    with psycopg.connect(db_uri_clean) as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT filename FROM thread_files WHERE thread_id = %s",
                                (tid,)
                            )
                            thread_files = [row[0] for row in cur.fetchall()]
                except Exception as db_err:
                    logger.error(f"[chat_node] DB query for thread files failed: {db_err}", exc_info=True)
                    
            if not thread_files:
                try:
                    vectorstore = get_vectorstore()
                    all_data = vectorstore.get(where={"thread_id": tid}, include=["metadatas"])
                    if all_data and "metadatas" in all_data and all_data["metadatas"]:
                        seen = set()
                        for m in all_data["metadatas"]:
                            if m and "filename" in m:
                                seen.add(m["filename"])
                            elif m and "source" in m:
                                seen.add(os.path.basename(m["source"]))
                        thread_files = list(seen)
                except Exception as vs_err:
                    logger.error(f"[chat_node] Vectorstore query for thread files failed: {vs_err}", exc_info=True)

        custom_system_prompt = SYSTEM_PROMPT
        if thread_files:
            files_str = ", ".join(f"'{f}'" for f in thread_files)
            custom_system_prompt += (
                f"\n\nCURRENT THREAD KNOWLEDGE BASE (UPLOADED/INDEXED DOCUMENTS):\n"
                f"The following document(s) are already indexed and available in the knowledge base for this session/thread: {files_str}.\n"
                f"When the user mentions 'uploaded pdf', 'this document', or asks questions about the uploaded PDF, they are referring to these file(s). "
                f"Use `search_knowledge_base` with these filenames directly to answer questions. Do NOT use index_local_file or search other files unless requested."
            )
        else:
            custom_system_prompt += (
                f"\n\nCURRENT THREAD KNOWLEDGE BASE (UPLOADED/INDEXED DOCUMENTS):\n"
                f"No documents are currently indexed in the knowledge base for this session/thread."
            )

        if summary:
            custom_system_prompt += (
                f"\n\nBACKGROUND CONTEXT (PREVIOUS CONVERSATION SUMMARY):\n"
                f"{summary}\n"
                f"Note: The above is a summary of the older part of the conversation for your memory. "
                f"Do NOT output, repeat, or refer to this summary in your response to the user. "
                f"Simply use it to maintain context."
            )

        custom_system_prompt += f"\n\nCRITICAL: You MUST strictly answer and converse in {language}. All replies, formatting, and answers must be in {language}."
        
        if not messages_to_send or not isinstance(messages_to_send[0], SystemMessage):
            messages_to_send = [SystemMessage(content=custom_system_prompt)] + messages_to_send
        else:
            messages_to_send[0] = SystemMessage(content=custom_system_prompt)

        response = invoke_llm_with_fallback(llm_with_tools, fallback_llm_with_tools, messages_to_send)
        
        if response and hasattr(response, "additional_kwargs"):
            if not response.additional_kwargs:
                response.additional_kwargs = {}
            if "timestamp" not in response.additional_kwargs:
                response.additional_kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return {
            "messages": [response],
            "language": language,
            "summary": summary,
            "last_summarized_count": last_summarized_count
        }

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", ToolNode(all_tools))
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile(checkpointer=checkpointer)
