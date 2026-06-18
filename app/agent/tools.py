import operator
import os
from typing import Literal, Annotated
import requests
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
from langchain_experimental.utilities import PythonREPL
from app.core.security import is_path_sensitive, is_code_safe, sanitize_value
from app.core.config import MCP_FS_ROOT
from app.core.logger import get_logger

logger = get_logger(__name__)

@tool
def web_search(query: str) -> str:
    """Search the web using DDGS."""
    from ddgs import DDGS
    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"Title: {r.get('title')}\nBody: {r.get('body')}\nLink: {r.get('href')}"
        for r in results
    )


@tool
def calculator(a: float, b: float, operation: Literal["+", "-", "*", "/"]) -> float:
    """Perform basic arithmetic operations."""
    ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
    if operation == "/" and b == 0:
        raise ValueError("Division by zero is not allowed.")
    return ops[operation](a, b)


@tool
def stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a given symbol."""
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_KEY not set in environment."}
    url = (
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
        f"&symbol={symbol}&apikey={api_key}"
    )
    return requests.get(url, timeout=20).json()


@tool
def write_file(filepath: str, content: str) -> str:
    """
    Write text content to a file on the local filesystem.
    Creates parent directories automatically if they don't exist.
    Use this whenever the user asks to save, create, or write a file.
    """
    abs_path = os.path.abspath(filepath)
    if not abs_path.startswith(MCP_FS_ROOT):
        return (
            f"Error: writing outside the allowed root '{MCP_FS_ROOT}' is not permitted. "
            f"Tried to write to '{abs_path}'."
        )
    if is_path_sensitive(abs_path):
        return f"Error: Access to path/file '{filepath}' is restricted."
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File saved successfully at: {abs_path}"


@tool
def read_file(filepath: str) -> str:
    """Read and return the text content of a file on the local filesystem."""
    abs_path = os.path.abspath(filepath)
    if not abs_path.startswith(MCP_FS_ROOT):
        return (
            f"Error: reading outside the allowed root '{MCP_FS_ROOT}' is not permitted. "
            f"Tried to read from '{abs_path}'."
        )
    if is_path_sensitive(abs_path):
        return f"Error: Access to path/file '{filepath}' is restricted."
    if not os.path.exists(abs_path):
        return f"Error: file not found at '{abs_path}'."

    # Prevent reading binary files as text
    _, ext = os.path.splitext(abs_path.lower())
    if ext in (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".odt", ".png", ".jpg", ".jpeg", ".gif", ".bin", ".db", ".sqlite", ".sqlite3", ".zip", ".tar", ".gz"):
        if ext == ".pdf":
            return (
                f"Error: '{filepath}' is a binary PDF file. You CANNOT read it as plain text using 'read_file'. "
                f"To query or get insights from this PDF document, you must use the 'search_knowledge_base' tool. "
                f"If the PDF is not already indexed in the knowledge base, you can index it first using 'index_local_file'."
            )
        elif ext in (".docx", ".doc", ".odt"):
            return (
                f"Error: '{filepath}' is a binary document. You CANNOT read it as plain text using 'read_file'. "
                f"To query or get insights from this document, it must be uploaded via the Knowledge Base sidebar panel in the UI."
            )
        elif ext in (".xlsx", ".xls"):
            return (
                f"Error: '{filepath}' is a binary Excel spreadsheet. You CANNOT read it as plain text using 'read_file'. "
                f"Instead, you must use the 'python_interpreter' tool to load the file (e.g. using pandas: `import pandas as pd; df = pd.read_excel('{filepath}')`) "
                f"and print the relevant sections/insights."
            )
        else:
            return f"Error: '{filepath}' is a binary file. Reading binary files as text is not supported."

    with open(abs_path, "r", encoding="utf-8") as f:
        return f.read()


@tool
def list_directory(path: str = ".") -> str:
    """List files and folders inside a directory. Defaults to the current working directory."""
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(MCP_FS_ROOT):
        return (
            f"Error: listing directories outside the allowed root '{MCP_FS_ROOT}' is not permitted. "
            f"Tried to list '{abs_path}'."
        )
    if is_path_sensitive(abs_path):
        return f"Error: Access to path/file '{path}' is restricted."
    if not os.path.isdir(abs_path):
        return f"Error: '{abs_path}' is not a directory."
    entries = os.listdir(abs_path)
    # Filter out sensitive entries
    entries = [e for e in entries if not is_path_sensitive(os.path.join(abs_path, e))]
    return "\n".join(sorted(entries)) if entries else "(empty directory)"


@tool
def python_interpreter(code: str) -> str:
    """
    A Python REPL. Use this to execute Python code, specifically for data analysis tasks (using pandas and numpy),
    calculations, or processing local files (like Excel and CSV sheets saved in the workspace).
    The code should print the final output or store it in a variable. The stdout of the code execution will be returned.
    Make sure to write clean, complete, executable Python code.
    """
    is_safe, error_msg = is_code_safe(code)
    if not is_safe:
        return f"Security Exception: {error_msg}"

    try:
        repl = PythonREPL()
        result = repl.run(code)
        return result if result else "Code executed successfully with no stdout output."
    except Exception as e:
        return f"Error executing Python code: {str(e)}"


@tool
def index_local_file(filepath: str, config: RunnableConfig) -> str:
    """
    Index a local PDF file from the workspace into the knowledge base.
    Use this when the user asks to analyze, summarize, or search a PDF file that is already in the workspace directory but not yet indexed.
    """
    try:
        from app.core.config import DB_URI
        from app.services.rag_pipeline import index_pdf_file
        
        abs_path = os.path.abspath(filepath)
        if not abs_path.startswith(MCP_FS_ROOT):
            return f"Error: File path '{filepath}' is outside the allowed root."
        if not os.path.exists(abs_path):
            return f"Error: File not found at '{filepath}'."
        if not abs_path.lower().endswith(".pdf"):
            return f"Error: Only PDF files can be indexed into the knowledge base."
            
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        filename = os.path.basename(abs_path)
        
        # Call index_pdf_file
        num_chunks = index_pdf_file(abs_path, filename, thread_id, DB_URI)
        return f"Successfully indexed '{filename}' into the knowledge base. Total chunks created: {num_chunks}. You can now query it using search_knowledge_base."
    except Exception as e:
        return f"Failed to index file: {str(e)}"


# Sync run helper for async tools (like MCP tools)
def make_sync_run(async_tool, main_loop_getter):
    def sync_run(*args, **kwargs):
        import asyncio

        tool_input = {}
        if args:
            schema = async_tool.args_schema
            if schema:
                if hasattr(schema, 'model_fields'):
                    fields = list(schema.model_fields.keys())
                else:
                    fields = list(schema.__fields__.keys())
                for i, arg in enumerate(args):
                    if i < len(fields):
                        tool_input[fields[i]] = arg
        if kwargs:
            tool_input.update(kwargs)

        # Security & Type checks: Intercept any input targeting sensitive paths or binary files in read operations
        is_read_tool = "read" in async_tool.name.lower()
        for k, v in tool_input.items():
            if isinstance(v, str):
                if is_path_sensitive(v):
                    err_msg = f"Security Exception: Access to path/file '{v}' is restricted."
                    if getattr(async_tool, 'response_format', 'content') == 'content_and_artifact':
                        return (err_msg, err_msg)
                    return err_msg

                if is_read_tool and k in ("path", "filepath"):
                    abs_path = os.path.abspath(v)
                    _, ext = os.path.splitext(abs_path.lower())
                    if ext in (".pdf", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".gif", ".bin", ".db", ".sqlite", ".sqlite3", ".zip"):
                        if ext == ".pdf":
                            err_msg = (
                                f"Error: '{v}' is a binary PDF file. You CANNOT read it as plain text using read tools. "
                                f"To retrieve information or get insights from this PDF document, you must use the 'search_knowledge_base' tool. "
                                f"If the PDF is not already indexed, you can index it using the 'index_local_file' tool."
                            )
                        elif ext in (".xlsx", ".xls"):
                            err_msg = (
                                f"Error: '{v}' is an Excel file. You CANNOT read it as plain text. "
                                f"Instead, use the 'python_interpreter' to write and execute python code (e.g. `import pandas as pd; print(pd.read_excel('{v}').head())`) to extract details."
                            )
                        else:
                            err_msg = f"Error: '{v}' is a binary file and cannot be read as text."
                        
                        if getattr(async_tool, 'response_format', 'content') == 'content_and_artifact':
                            return (err_msg, err_msg)
                        return err_msg

        # Force headless=True for playwright tools if not explicitly overridden by LLM
        if "playwright" in async_tool.name:
            schema = async_tool.args_schema
            if schema:
                if hasattr(schema, 'model_fields'):
                    has_headless = 'headless' in schema.model_fields
                else:
                    has_headless = 'headless' in getattr(schema, '__fields__', {})
                if has_headless and 'headless' not in tool_input:
                    tool_input['headless'] = True

        try:
            loop = main_loop_getter()
            if loop is not None and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(async_tool.coroutine(**tool_input), loop)
                res = future.result()
            else:
                res = asyncio.run(async_tool.coroutine(**tool_input))
            
            # Sanitize return output recursively if it's a filesystem tool
            if "filesystem" in async_tool.name:
                res = sanitize_value(res)
            return res
        except Exception as e:
            # Return execution error message in the format expected by the tool
            err_msg = f"Error executing tool: {str(e)}"
            if getattr(async_tool, 'response_format', 'content') == 'content_and_artifact':
                return (err_msg, err_msg)
            return err_msg

    return sync_run


@tool("playwright_playwright_navigate")
def playwright_playwright_navigate(
    url: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    **kwargs
) -> str:
    """
    Navigate to a URL using Playwright and perform ephemeral web page RAG.
    Use this to retrieve and analyze the full content, charts, and images of any web page.
    """
    import hashlib
    import os
    import shutil
    import asyncio
    import threading
    from app.services.rag_pipeline import index_pdf_file, _search_knowledge_base_logic, get_vectorstore, invalidate_bm25_cache, sanitize_filename
    
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    
    # Generate hashed filename
    hasher = hashlib.md5()
    hasher.update(f"{url}_{thread_id}".encode("utf-8"))
    hash_str = hasher.hexdigest()[:12]
    filename = f"web_{hash_str}.pdf"
    temp_pdf_path = os.path.join("/tmp", filename)
    
    # Extract query from conversation state
    messages = state.get("messages", [])
    user_query = ""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            user_query = str(msg.content)
            break
            
    if not user_query:
        user_query = f"Explain the page content of {url}"
        
    logger.info(f"[Playwright RAG] URL: {url} | thread: {thread_id} | query: '{user_query}'")
    
    # Run playwright in a separate event loop on a dedicated thread to be 100% safe
    def run_playwright_pdf():
        from playwright.async_api import async_playwright
        
        async def capture():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                try:
                    # wait_until='networkidle' with a timeout=15000 (15 seconds) fallback
                    await page.goto(url, wait_until="networkidle", timeout=15000)
                except Exception as e:
                    logger.warning(f"[Playwright RAG] Navigation timeout or error: {e}")
                
                # Capture the PDF of the full page scroll
                await page.pdf(path=temp_pdf_path, print_background=True)
                await browser.close()
                
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(capture())
        finally:
            loop.close()
            
    t = threading.Thread(target=run_playwright_pdf)
    t.start()
    t.join()
    
    if not os.path.exists(temp_pdf_path) or os.path.getsize(temp_pdf_path) == 0:
        return f"Error: Failed to capture PDF of the web page {url}."
        
    # Index the PDF file ephemerally (db_uri = None)
    chunks_added = index_pdf_file(
        temp_path=temp_pdf_path,
        filename=filename,
        thread_id=thread_id,
        db_uri=None
    )
    
    logger.info(f"[Playwright RAG] Indexed {filename} ({chunks_added} chunks).")
    
    # Query the data using the internal logic directly bypassing the @tool decorator wrapper
    result_text = ""
    try:
        # Determine if the user explicitly asks for visual or image analysis
        query_lower = user_query.lower()
        is_visual_query = any(k in query_lower for k in ("image", "chart", "graph", "diagram", "describe", "look at", "picture", "visual", "plot"))
        text_only_val = not is_visual_query
        
        result_text = _search_knowledge_base_logic(
            query=user_query,
            thread_id=thread_id,
            filename=filename,
            text_only=text_only_val
        )
    except Exception as query_err:
        logger.error(f"[Playwright RAG] Query error: {query_err}", exc_info=True)
        result_text = f"Error performing search: {query_err}"
        
    # Clean up (CRITICAL ORDER)
    try:
        # 1. Delete all ChromaDB chunks where thread_id == current AND filename == hashed_filename
        vectorstore = get_vectorstore()
        vectorstore.delete(where={
            "$and": [
                {"thread_id": {"$eq": thread_id}},
                {"filename": {"$eq": filename}}
            ]
        })
        invalidate_bm25_cache(thread_id)
        
        # 2. Delete the /tmp/ PDF file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            
        # 3. Delete saved images
        sanitized_hashed_filename = sanitize_filename(filename)
        images_dir = os.path.join("static", "extracted_images", sanitized_hashed_filename)
        shutil.rmtree(images_dir, ignore_errors=True)
        
        logger.info(f"[Playwright RAG] Ephemeral cleanup done for {filename}.")
    except Exception as cleanup_err:
        logger.error(f"[Playwright RAG] Cleanup error: {cleanup_err}", exc_info=True)
        
    return result_text
