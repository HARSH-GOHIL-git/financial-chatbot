import os
import re
import traceback
import json
import hashlib
import sqlite3
import PIL.Image
from google import genai
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.db import get_db_connection

from app.core.logger import get_logger
logger = get_logger(__name__)


_cross_encoder = None
_embeddings = None
_vectorstore = None
import threading
_bm25_lock = threading.Lock()
_bm25_indices = {}  # Thread-scoped BM25 indices: thread_id -> (bm25, all_docs)

# SQLite Cache DB Path
DB_CACHE_PATH = "image_cache.db"

def init_cache_db():
    """Initializes the SQLite cache database for image descriptions."""
    try:
        conn = sqlite3.connect(DB_CACHE_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_descriptions (
                image_hash TEXT PRIMARY KEY,
                description TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"[Cache] Failed to initialise SQLite cache: {e}", exc_info=True)

# Initialize the cache database
init_cache_db()

def get_cached_description(image_hash: str) -> str:
    """Retrieves a cached description for a given image hash, if available."""
    try:
        with sqlite3.connect(DB_CACHE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT description FROM image_descriptions WHERE image_hash = ?", (image_hash,))
            row = cursor.fetchone()
            if row:
                return row[0]
    except Exception as e:
        logger.warning(f"[Cache] Query failed: {e}")
    return None

def save_to_cache(image_hash: str, description: str):
    """Saves an image description to the SQLite cache."""
    try:
        with sqlite3.connect(DB_CACHE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO image_descriptions (image_hash, description, timestamp)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (image_hash, description))
            conn.commit()
    except Exception as e:
        logger.warning(f"[Cache] Write failed: {e}")

def call_gemini_flash_vlm(image_path: str, query: str) -> str:
    """Calls Gemini 2.5 Flash (falling back to gemini-3.1-flash-lite if needed) to analyze an image."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "[Error: GOOGLE_API_KEY not set in environment]"
            
        client = genai.Client(api_key=api_key)
        image = PIL.Image.open(image_path)
        prompt = (
            f"You are analyzing an image extracted from a document.\n"
            f"Focus specifically on answering the user's question: \"{query}\"\n\n"
            f"Please describe the image. Be detailed and explain in no more than 300-400 words. In particular:\n"
            f"- Extract all visible numbers, labels, trends, and axes.\n"
            f"- Focus specifically on what is relevant to the user's question.\n"
            f"- Do not assume or extrapolate beyond what is clearly visible."
        )
        
        # Try gemini-2.5-flash first, fallback to gemini-3.1-flash-lite on failure (due to 503/429)
        for model_name in ('gemini-2.5-flash', 'gemini-3.1-flash-lite'):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[image, prompt]
                )
                if response.text:
                    return response.text
            except Exception as inner_err:
                logger.warning(f"[VLM] Model {model_name} failed: {inner_err}")
                continue
        raise Exception("All VLM models failed to generate content.")
    except Exception as e:
        logger.error(f"[VLM] Gemini VLM call failed: {e}", exc_info=True)
        return f"[Error analyzing image at {image_path}: {str(e)}]"

# ─────────────────────────────────────────────────────────────
# IMAGE INGESTION: RapidOCR
# Used during PDF indexing to extract visible text from images.
# At query time, Gemini Flash VLM is used for deeper analysis.
# ─────────────────────────────────────────────────────────────
_rapid_ocr_instance = None
_rapid_ocr_lock = threading.Lock()

def get_rapid_ocr():
    global _rapid_ocr_instance
    with _rapid_ocr_lock:
        if _rapid_ocr_instance is None:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_ocr_instance = RapidOCR()
        return _rapid_ocr_instance

def process_image_content(image_path: str) -> str:
    """Extracts text from an image using RapidOCR during document ingestion."""
    try:
        engine = get_rapid_ocr()
        result, elapse = engine(image_path)
        if result:
            texts = [line[1] for line in result]
            return "\n".join(texts).strip()
    except Exception as e:
        logger.warning(f"[RapidOCR] Failed on {image_path}: {e}")
    return ""



def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        logger.info("[Models] Loading CrossEncoder 'BAAI/bge-reranker-base'...")
        _cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    return _cross_encoder

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        logger.info("[Models] Loading HuggingFaceEmbeddings 'BAAI/bge-base-en-v1.5'...")
        _embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return _embeddings

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        logger.info("[Models] Loading Chroma vectorstore from './chroma_db'...")
        _vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=get_embeddings())
    return _vectorstore

import re as _re
_bib_pattern = _re.compile(r'^\s*\[[A-Za-z+\d]{3,10}\]')

def is_bibliography_chunk(text: str) -> bool:
    return bool(_bib_pattern.match(text)) or len(text.strip()) < 150

STOPWORDS = {
    "a","an","the","is","it","in","on","of","to","and",
    "or","for","with","that","this","as","by","at","from",
    "are","was","were","be","been","has","have","had","not"
}

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def get_bm25_index(vectorstore, thread_id: str):
    global _bm25_indices
    with _bm25_lock:
        if thread_id not in _bm25_indices:
            all_data = vectorstore.get(where={"thread_id": thread_id})
            if not all_data or "documents" not in all_data or not all_data["documents"]:
                return None
            all_docs = []
            for doc_text, metadata in zip(all_data["documents"], all_data["metadatas"] or []):
                all_docs.append(Document(page_content=doc_text, metadata=metadata or {}))
                
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [tokenize(doc.page_content) for doc in all_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Enforce 50-entry cache limit to avoid memory leak risks
            if len(_bm25_indices) >= 50:
                oldest_key = next(iter(_bm25_indices))
                del _bm25_indices[oldest_key]
                logger.warning(f"[BM25 Cache] Evicted entry for thread '{oldest_key}' (50-entry limit reached).")
                
            _bm25_indices[thread_id] = (bm25, all_docs)
        return _bm25_indices[thread_id]

def invalidate_bm25_cache(thread_id: str):
    global _bm25_indices
    with _bm25_lock:
        if thread_id in _bm25_indices:
            del _bm25_indices[thread_id]


def expand_query(original_query: str) -> list[str]:
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return [original_query]
            
        client = genai.Client(api_key=api_key)
        prompt = f"""Generate 3 alternative search queries for retrieving relevant 
    chunks from a financial document for this question: '{original_query}'
    Return only a JSON array of 3 strings, nothing else."""
        
        # Try gemini-3.1-flash-lite first, fallback to gemini-2.5-flash on failure
        llm_response = None
        for model_name in ('gemini-3.1-flash-lite','gemini-2.5-flash'):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response.text:
                    llm_response = response.text
                    break
            except Exception as inner_err:
                logger.warning(f"[Query Expansion] Model {model_name} failed: {inner_err}")
                continue
                
        if not llm_response:
            return [original_query]
            
        # Strip any markdown code blocks if the LLM returned it wrapped in ```json ... ```
        clean_response = llm_response.strip()
        if clean_response.startswith("```"):
            lines = clean_response.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_response = "\n".join(lines).strip()
            
        variants = json.loads(clean_response)
        if isinstance(variants, list):
            return [original_query] + [str(v) for v in variants[:3]]
        return [original_query]
    except Exception as e:
        logger.error(f"[Query Expansion] Failed to expand query: {e}", exc_info=True)
        return [original_query]


@tool
def search_knowledge_base(
    query: str,
    config: RunnableConfig,
    filename: str = None,
    page: int = None,
    text_only: bool = False
) -> str:
    """
    Search the local RAG knowledge base for context.
    Use this when the user asks questions about specific documents or uploaded PDFs.

    Set text_only=True for:
    - Factual queries (prices, dates, names, statistics)
    - Web page captures where answer is likely in text
    - Any query where the user has NOT explicitly asked to look at, analyze, or describe an image, chart, or diagram

    Set text_only=False (default) when:
    - User explicitly asks to analyze, describe, or look at a chart, image, graph, or diagram
    - Text chunks returned insufficient context and visual data may help
    - Query is about document structure, layout, or visual content

    If the user provides a specific page number, pass the 1-indexed page number via the 'page' parameter.
    If the user references a specific file, pass it via the 'filename' parameter.
    """
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
        vectorstore = get_vectorstore()
        
        # 1. Fetch files associated with this thread to match/resolve the filename
        thread_files = []
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT filename FROM thread_files WHERE thread_id = %s",
                        (thread_id,)
                    )
                    thread_files = [row[0] for row in cur.fetchall()]
        except Exception as db_err:
            logger.error(f"[RAG] DB query for thread files failed: {db_err}", exc_info=True)
                
        # If DB query failed or empty, fallback to querying Chroma unique filenames
        if not thread_files:
            try:
                all_data = vectorstore.get(where={"thread_id": thread_id}, include=["metadatas"])
                if all_data and "metadatas" in all_data and all_data["metadatas"]:
                    seen = set()
                    for m in all_data["metadatas"]:
                        if m and "filename" in m:
                            seen.add(m["filename"])
                        elif m and "source" in m:
                            seen.add(os.path.basename(m["source"]))
                    thread_files = list(seen)
            except Exception as vs_err:
                logger.error(f"[RAG] Vectorstore query for thread files failed: {vs_err}", exc_info=True)

        # If still no documents, we can't proceed
        if not thread_files:
            return "The knowledge base is empty for this thread. Please upload a PDF first."

        # Match filename if provided
        matched_filename = None
        if filename:
            fn_lower = filename.lower()
            # 1. Exact case-insensitive match
            for f in thread_files:
                if f.lower() == fn_lower:
                    matched_filename = f
                    break
            # 2. Substring match
            if not matched_filename:
                for f in thread_files:
                    if fn_lower in f.lower() or f.lower() in fn_lower:
                        matched_filename = f
                        break
            # 3. Fallback
            if not matched_filename:
                matched_filename = filename
        else:
            # If filename is not provided, and there's only one file in the thread, default to it
            if len(thread_files) == 1:
                matched_filename = thread_files[0]
            else:
                # If there are multiple files, and no specific page is requested, we will search across all.
                # But if a specific page IS requested, we must know which file it is!
                if page is not None:
                    return f"Multiple documents are indexed in this thread: {thread_files}. Please specify the 'filename' parameter in your search."

        # If a specific page is requested, retrieve all chunks for that page directly
        if page is not None:
            if page <= 0:
                return f"Invalid page number {page}. Page numbers must be greater than or equal to 1."
            
            page_num = page - 1
            where_clause = {
                "$and": [
                    {"thread_id": {"$eq": thread_id}},
                    {"filename": {"$eq": matched_filename}},
                    {"page": {"$eq": page_num}}
                ]
            }
            
            res = vectorstore.get(where=where_clause)
            
            if not res or not res.get("documents"):
                # Check if the file is indexed at all
                file_where = {
                    "$and": [
                        {"thread_id": {"$eq": thread_id}},
                        {"filename": {"$eq": matched_filename}}
                    ]
                }
                file_res = vectorstore.get(where=file_where, include=["metadatas"])
                if not file_res or not file_res.get("documents"):
                    return f"The file '{matched_filename}' is not indexed in this thread. Available files: {thread_files}."
                
                # Find maximum page number
                pages = [m.get("page", 0) for m in file_res["metadatas"] if m]
                max_page = max(pages) + 1 if pages else 0
                return f"Page {page} not found in '{matched_filename}'. The document has {max_page} page(s)."
            
            text_table_parts = []
            image_descriptions = []
            
            for i, doc_text in enumerate(res["documents"]):
                meta = res["metadatas"][i]
                chunk_type = meta.get("chunk_type", "text")
                
                if chunk_type in ("text", "table"):
                    text_table_parts.append(doc_text)
                elif chunk_type == "image":
                    if text_only:
                        continue
                    image_path = meta.get("image_path")
                    image_hash = meta.get("image_hash")
                    
                    description = None
                    # First check the cache if image_hash is available
                    if image_hash:
                        description = get_cached_description(image_hash)
                    
                    if not description:
                        # Call VLM directly as requested if not cached
                        if image_path and os.path.exists(image_path):
                            vlm_prompt = f"Explain this image from page {page} of '{matched_filename}' in detail."
                            if query and query.strip():
                                vlm_prompt = f"Explain this image in 300-400 words from page {page} of '{matched_filename}' in the context of the question: '{query}'"
                            
                            description = call_gemini_flash_vlm(image_path, vlm_prompt)
                            if image_hash and description and not description.startswith("[Error"):
                                save_to_cache(image_hash, description)
                        else:
                            description = f"[Image file not found at {image_path}]"
                    image_descriptions.append(description)
                    
            output_parts = [f"=== Content of Page {page} of {matched_filename} ==="]
            if text_table_parts:
                joined_text = "\n\n".join(text_table_parts)
                output_parts.append(f"--- Text & Tables ---\n{joined_text}")
            if image_descriptions:
                output_parts.append("--- Image Analysis (VLM) ---")
                for idx, desc in enumerate(image_descriptions):
                    output_parts.append(f"[Image {idx+1}]: {desc}")
                    
            return "\n\n".join(output_parts)

        # Standard hybrid search flow across vector and BM25 (filtered by matched_filename if provided)
        search_filter = {"thread_id": thread_id}
        if matched_filename:
            search_filter = {
                "$and": [
                    {"thread_id": {"$eq": thread_id}},
                    {"filename": {"$eq": matched_filename}}
                ]
            }

        # Expand query: original query + 3 variants
        queries = expand_query(query)
        logger.info(f"[RAG] Expanded queries: {queries}")

        # 1. Retrieve from Vector Search for all queries
        vector_docs_lists = []
        for q in queries:
            q_vector_docs = vectorstore.similarity_search(q, k=15, filter=search_filter)
            vector_docs_lists.append(q_vector_docs)
            
        # 2. Retrieve from BM25 Search for all queries
        bm25_docs_lists = []
        bm25_info = get_bm25_index(vectorstore, thread_id)
        if bm25_info:
            bm25, all_docs = bm25_info
            # Filter BM25 corpus by filename if matched_filename is active
            if matched_filename:
                filtered_docs = [doc for doc in all_docs if doc.metadata.get("filename") == matched_filename]
            else:
                filtered_docs = all_docs
                
            filtered_docs = [
                doc for doc in filtered_docs 
                if not is_bibliography_chunk(doc.page_content)
            ]
                
            if filtered_docs:
                from rank_bm25 import BM25Okapi
                tokenized_corpus = [tokenize(doc.page_content) for doc in filtered_docs]
                file_bm25 = BM25Okapi(tokenized_corpus)
                for q in queries:
                    tokenized_query = tokenize(q)
                    doc_scores = file_bm25.get_scores(tokenized_query)
                    bm25_results = sorted(zip(filtered_docs, doc_scores), key=lambda x: x[1], reverse=True)
                    q_bm25_top = [doc for doc, score in bm25_results[:15] if score > 0]
                    bm25_docs_lists.append(q_bm25_top)
            else:
                bm25_docs_lists = [[] for _ in queries]
        else:
            bm25_docs_lists = [[] for _ in queries]
            
        # 3. Reciprocal Rank Fusion (RRF) to merge Vector and BM25 results across all queries
        def reciprocal_rank_fusion(vector_lists, bm25_lists, k=60):
            rrf_scores = {}
            doc_map = {}
            
            for vector_docs in vector_lists:
                for rank, doc in enumerate(vector_docs):
                    content = doc.page_content
                    doc_map[content] = doc
                    rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k + rank + 1)
                    
            for bm25_docs in bm25_lists:
                for rank, doc in enumerate(bm25_docs):
                    content = doc.page_content
                    doc_map[content] = doc
                    rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k + rank + 1)
                    
            sorted_contents = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            return [doc_map[content] for content in sorted_contents]
            
        merged_docs = reciprocal_rank_fusion(vector_docs_lists, bm25_docs_lists)
        candidates = merged_docs[:30]
        
        if not candidates:
            return "No relevant information found in the knowledge base."
            
        # 4. Cross-Encoder Re-ranking
        cross_encoder = get_cross_encoder()
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = cross_encoder.predict(pairs)
        
        ranked_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in ranked_candidates[:6]]
        
        # Format the results and resolve lazy VLM for images
        context_parts = []
        for i, doc in enumerate(top_docs):
            filename_doc = doc.metadata.get('filename', 'Unknown Document')
            page_num = doc.metadata.get('page', 0) + 1
            chunk_type = doc.metadata.get('chunk_type', 'text')
            
            if chunk_type in ('text', 'table'):
                context_parts.append(
                    f"--- Excerpt {i+1} (Source: {filename_doc}, Page: {page_num}) ---\n{doc.page_content}"
                )
            elif chunk_type == 'image':
                if text_only:
                    continue
                image_path = doc.metadata.get('image_path')
                image_hash = doc.metadata.get('image_hash')
                
                needs_vlm = doc.metadata.get("needs_vlm")
                vlm_done = doc.metadata.get("vlm_done")
                
                # Backward compatibility for older records:
                if needs_vlm is None:
                    if image_hash and get_cached_description(image_hash):
                        needs_vlm = False
                        vlm_done = True
                    else:
                        needs_vlm = True
                        vlm_done = False
                
                description = None
                if needs_vlm == True and vlm_done == False:
                    # Call existing Gemini Flash VLM function on this image
                    if image_path and os.path.exists(image_path):
                        description = call_gemini_flash_vlm(image_path, query)
                        # Cache the Gemini output in SQLite using existing cache logic
                        if image_hash and description and not description.startswith("[Error"):
                            save_to_cache(image_hash, description)
                    else:
                        description = f"[Image file not found at {image_path}]"
                    
                    # Replace chunk text with Gemini output in-memory only
                    doc.page_content = description
                    # Mark vlm_done=True in-memory for this request only
                    doc.metadata["vlm_done"] = True
                
                elif needs_vlm == False and vlm_done == True:
                    # Use stored description directly, no Gemini call needed
                    description = doc.page_content
                else:
                    description = doc.page_content
                
                context_parts.append(
                    f"--- Excerpt {i+1} [Image Analysis] (Source: {filename_doc}, Page: {page_num}) ---\n{description}"
                )
        
        return "\n\n".join(context_parts)
    except Exception as e:
        logger.error(f"[RAG] search_knowledge_base failed: {e}", exc_info=True)
        return f"Failed to retrieve from knowledge base: {str(e)}"

def _search_knowledge_base_logic(
    query: str,
    thread_id: str,
    filename: str = None,
    page: int = None,
    text_only: bool = False
) -> str:
    """Internal function to search the knowledge base bypassing the LangChain @tool wrapper."""
    config = {"configurable": {"thread_id": thread_id}}
    return search_knowledge_base.func(query, config, filename, page, text_only=text_only)

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\.-]', '_', name)

def index_pdf_file(temp_path: str, filename: str, thread_id: str, db_uri: str = None) -> int:
    """Chunks PDF and extracts text, tables, and images, storing them in ChromaDB."""
    import fitz
    
    doc = fitz.open(temp_path)
    all_chunks = []

    # Prepare image saving directory
    sanitized_doc_name = sanitize_filename(filename)
    images_dir = os.path.join("static", "extracted_images", sanitized_doc_name)
    os.makedirs(images_dir, exist_ok=True)

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 1. Find tables
        tables_finder = page.find_tables()
        tables = tables_finder.tables if tables_finder else []
        table_rects = [fitz.Rect(t.bbox) for t in tables]
        
        for table in tables:
            markdown_table = table.to_markdown()
            if markdown_table.strip():
                table_chunk = Document(
                    page_content=markdown_table,
                    metadata={
                        "thread_id": thread_id,
                        "filename": filename,
                        "page": page_num,
                        "chunk_type": "table"
                    }
                )
                all_chunks.append(table_chunk)
                
        # 2. Extract images
        image_list = page.get_images(full=True)
        seen_xrefs = set()
        for img_info in image_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            
            try:
                base_image = doc.extract_image(xref)
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_width = base_image.get("width", 0)
                    image_height = base_image.get("height", 0)
                    image_size = len(image_bytes)
                    
                    # Filter out small images (logos, icons, signatures, etc.)
                    # Thresholds: width < 150px, height < 150px, area < 50,000px^2, file size < 20KB, or aspect_ratio > 3.0 (wide/narrow banners)
                    area = image_width * image_height
                    aspect_ratio = max(image_width, image_height) / max(min(image_width, image_height), 1)
                    if image_width < 150 or image_height < 150 or area < 50000 or image_size < 20 * 1024 or aspect_ratio > 3.0:
                        print(f"[Ingestion] Skipping decorative image xref {xref} (w={image_width}, h={image_height}, size={image_size}B, ar={aspect_ratio:.2f})")
                        continue
                        
                    image_hash = hashlib.sha256(image_bytes).hexdigest()
                    
                    # Save cropped image section where possible, fallback to raw image
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        image_filename = f"img_page_{page_num}_{xref}.png"
                        image_path = os.path.join(images_dir, image_filename)
                        img_rect = img_rects[0]
                        pix = page.get_pixmap(clip=img_rect, dpi=150)
                        pix.save(image_path)
                    else:
                        image_filename = f"img_page_{page_num}_{xref}.{image_ext}"
                        image_path = os.path.join(images_dir, image_filename)
                        with open(image_path, "wb") as f_img:
                            f_img.write(image_bytes)
                    
                    # Step 2 — SQLite cache check:
                    cached_desc = get_cached_description(image_hash)
                    if cached_desc:
                        image_chunk = Document(
                            page_content=cached_desc,
                            metadata={
                                "thread_id": thread_id,
                                "filename": filename,
                                "page": page_num,
                                "chunk_type": "image",
                                "image_path": image_path,
                                "image_hash": image_hash,
                                "needs_vlm": False,
                                "vlm_done": True
                            }
                        )
                        all_chunks.append(image_chunk)
                    else:
                        # Step 3 — Surrounding context & Case A/B branching:
                        # Bounding box & surrounding context
                        if img_rects:
                            img_rect = img_rects[0]
                            # Extend bounding box vertically by 200 points
                            extended_rect = fitz.Rect(max(0, img_rect.x0 - 50), max(0, img_rect.y0 - 150), min(page.rect.width, img_rect.x1 + 50), min(img_rect.y1 + 150, page.rect.height))
                        else:
                            extended_rect = page.rect
                        
                        # Extract surrounding text from text blocks on the page
                        blocks = page.get_text("blocks")
                        surrounding_blocks = []
                        for b in blocks:
                            if b[6] == 0:  # Text block
                                block_rect = fitz.Rect(b[:4])
                                # Exclude text in tables
                                if any(block_rect.intersects(tr) for tr in table_rects):
                                    continue
                                if block_rect.intersects(extended_rect):
                                    surrounding_blocks.append(b[4])
                        
                        surrounding_text = "\n".join(surrounding_blocks).strip()
                        has_surrounding = bool(surrounding_text.strip())
                        
                        smolvlm_success = False
                        smolvlm_output = ""
                        try:
                            smolvlm_output = process_image_content(image_path)
                            if smolvlm_output:
                                smolvlm_success = True
                        except Exception as e:
                            logger.warning(f"[Ingestion] Visual processing failed on {image_path}: {e}")
                        
                        if has_surrounding:
                            # CASE A
                            if smolvlm_success:
                                chunk_text = f"{surrounding_text}\n[Visual hint: {smolvlm_output}]"
                            else:
                                chunk_text = surrounding_text
                        else:
                            # CASE B
                            if smolvlm_success:
                                chunk_text = smolvlm_output
                            else:
                                chunk_text = ""
                        
                        if smolvlm_success and image_hash:
                            save_to_cache(image_hash, smolvlm_output)
                            
                        image_chunk = Document(
                            page_content=chunk_text,
                            metadata={
                                "thread_id": thread_id,
                                "filename": filename,
                                "page": page_num,
                                "chunk_type": "image",
                                "image_path": image_path,
                                "image_hash": image_hash,
                                "needs_vlm": not smolvlm_success,
                                "vlm_done": smolvlm_success
                            }
                        )
                        all_chunks.append(image_chunk)
            except Exception as img_err:
                logger.error(f"[Ingestion] Error extracting image xref {xref} on page {page_num}: {img_err}", exc_info=True)
                
        # 2.5 Detect vector drawings and take a page snapshot if necessary
        try:
            paths = page.get_drawings()
            # Only count paths that are NOT inside any table bounding box
            non_table_paths = [
                p for p in paths
                if not any(p["rect"].intersects(tr) for tr in table_rects)
            ]
            
            # Filter out decorative/layout drawings (headers, footers, full page borders)
            content_paths = []
            for p in non_table_paths:
                r = p["rect"]
                # Skip if empty rect
                if r.is_empty:
                    continue
                # Skip if it is a full page border or very thin line spanning most of page width/height
                if (r.width > page.rect.width * 0.9 and r.height < 10) or (r.height > page.rect.height * 0.9 and r.width < 10):
                    continue
                # Skip if it spans the whole page size (like background or page border rect)
                if r.width > page.rect.width * 0.95 and r.height > page.rect.height * 0.95:
                    continue
                # Skip if in header region (top 80 pt)
                if r.y1 < 80:
                    continue
                # Skip if in footer region (bottom 80 pt from the end)
                if r.y0 > page.rect.height - 80:
                    continue
                content_paths.append(p)
                
            if len(content_paths) > 10:
                # Calculate bounding box of all content paths
                bbox = fitz.Rect(content_paths[0]["rect"])
                for p in content_paths[1:]:
                    bbox.include_rect(p["rect"])
                
                # Check if we have a valid, non-empty bounding box
                if not bbox.is_empty and bbox.width > 20 and bbox.height > 20:
                    # Pad it slightly to ensure we capture the whole drawing nicely (e.g. 15 pt)
                    padding = 15
                    bbox.x0 = max(0, bbox.x0 - padding)
                    bbox.y0 = max(0, bbox.y0 - padding)
                    bbox.x1 = min(page.rect.width, bbox.x1 + padding)
                    bbox.y1 = min(page.rect.height, bbox.y1 + padding)
                    
                    # Render only this section of the page as a snapshot
                    pix = page.get_pixmap(clip=bbox, dpi=150)
                    page_img_filename = f"page_{page_num}_snapshot.png"
                    page_img_path = os.path.join(images_dir, page_img_filename)
                    pix.save(page_img_path)
                    
                    # Calculate image hash for caching
                    with open(page_img_path, "rb") as f_img:
                        page_img_hash = hashlib.sha256(f_img.read()).hexdigest()
                    
                    # Step 2 — SQLite cache check:
                    cached_desc = get_cached_description(page_img_hash)
                    if cached_desc:
                        page_chunk = Document(
                            page_content=cached_desc,
                            metadata={
                                "thread_id": thread_id,
                                "filename": filename,
                                "page": page_num,
                                "chunk_type": "image",
                                "image_path": page_img_path,
                                "image_hash": page_img_hash,
                                "needs_vlm": False,
                                "vlm_done": True
                            }
                        )
                        all_chunks.append(page_chunk)
                    else:
                        # Step 3 — CASE A/B branching:
                        # Bounding box & surrounding context for vector drawings
                        extended_rect = fitz.Rect(max(0, bbox.x0 - 50), max(0, bbox.y0 - 150), min(page.rect.width, bbox.x1 + 50), min(bbox.y1 + 150, page.rect.height))
                        
                        # Extract surrounding text from text blocks on the page
                        blocks = page.get_text("blocks")
                        surrounding_blocks = []
                        for b in blocks:
                            if b[6] == 0:  # Text block
                                block_rect = fitz.Rect(b[:4])
                                # Exclude text in tables
                                if any(block_rect.intersects(tr) for tr in table_rects):
                                    continue
                                if block_rect.intersects(extended_rect):
                                    surrounding_blocks.append(b[4])
                        
                        surrounding_text = "\n".join(surrounding_blocks).strip()
                        has_surrounding = bool(surrounding_text.strip())
                        
                        smolvlm_success = False
                        smolvlm_output = ""
                        try:
                            smolvlm_output = process_image_content(page_img_path)
                            if smolvlm_output:
                                smolvlm_success = True
                        except Exception as e:
                            logger.warning(f"[Ingestion] Visual processing failed on snapshot {page_img_path}: {e}")
                        
                        if has_surrounding:
                            # CASE A
                            if smolvlm_success:
                                chunk_text = f"{surrounding_text}\n[Visual hint: {smolvlm_output}]"
                            else:
                                chunk_text = surrounding_text
                        else:
                            # CASE B
                            if smolvlm_success:
                                chunk_text = smolvlm_output
                            else:
                                chunk_text = ""
                        
                        if smolvlm_success and page_img_hash:
                            save_to_cache(page_img_hash, smolvlm_output)
                            
                        page_chunk = Document(
                            page_content=chunk_text,
                            metadata={
                                "thread_id": thread_id,
                                "filename": filename,
                                "page": page_num,
                                "chunk_type": "image",
                                "image_path": page_img_path,
                                "image_hash": page_img_hash,
                                "needs_vlm": not smolvlm_success,
                                "vlm_done": smolvlm_success
                            }
                        )
                        all_chunks.append(page_chunk)
                    logger.info(f"[Ingestion] Page {page_num + 1}: vector drawing snapshot saved ({bbox.width:.0f}x{bbox.height:.0f}px) at {page_img_path}.")
        except Exception as snap_err:
            logger.error(f"[Ingestion] Snapshot failed on page {page_num}: {snap_err}", exc_info=True)
            
        # 3. Extract text excluding table areas
        blocks = page.get_text("blocks")
        text_parts = []
        for b in blocks:
            if b[6] == 0:  # Text block
                rect = fitz.Rect(b[:4])
                if any(rect.intersects(tr) for tr in table_rects):
                    continue
                text_parts.append(b[4])
        
        page_text = "\n".join(text_parts).strip()
        if page_text:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            splits = splitter.split_text(page_text)
            for split_text in splits:
                if split_text.strip():
                    text_chunk = Document(
                        page_content=split_text,
                        metadata={
                            "thread_id": thread_id,
                            "filename": filename,
                            "page": page_num,
                            "chunk_type": "text"
                        }
                    )
                    all_chunks.append(text_chunk)

    vectorstore = get_vectorstore()
    if all_chunks:
        vectorstore.add_documents(all_chunks)

        # Debug hook (disabled in production)
        # try:
        #     from inspect_chunks import save_chunks_to_disk
        #     save_chunks_to_disk(all_chunks, filename)
        # except Exception as debug_err:
        #     print(f"[Debug Chunks] Failed to auto-save chunks to disk: {debug_err}")

    # Invalidate the BM25 cache since new documents are added
    invalidate_bm25_cache(thread_id)

    # Register in thread_files table
    if db_uri:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO thread_files (thread_id, filename, file_type)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (thread_id, filename) DO NOTHING;
                    """, (thread_id, filename, 'pdf'))
                    conn.commit()
        except Exception as db_err:
            logger.error(f"[RAG] DB register for PDF failed: {db_err}", exc_info=True)
    return len(all_chunks)


def convert_docx_to_pdf(docx_path: str) -> str:
    """Converts a DOCX file to PDF using headless LibreOffice."""
    import subprocess
    
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to", "pdf",
        "--outdir", "/tmp",
        docx_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LibreOffice conversion failed: {e.stderr or e.stdout}")
        
    # The output PDF filename will have the same base name as the DOCX file
    base_name = os.path.splitext(os.path.basename(docx_path))[0]
    pdf_path = os.path.join("/tmp", f"{base_name}.pdf")
    
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        raise RuntimeError(f"Converted PDF file is missing or empty: {pdf_path}")
        
    return pdf_path


def index_docx_file(temp_path: str, filename: str, thread_id: str, db_uri: str = None) -> int:
    """Converts DOCX to PDF first, then indexes the PDF, registering as docx."""
    pdf_path = None
    try:
        # 1. Call convert_docx_to_pdf(temp_path)
        pdf_path = convert_docx_to_pdf(temp_path)
        
        # 2. Call the existing index_pdf_file() with that PDF path, using db_uri=None so it doesn't write 'pdf' to db
        chunks_added = index_pdf_file(pdf_path, filename, thread_id, db_uri=None)
        
        # 3. Register file_type = 'docx' in the thread_files table
        if db_uri:
            try:
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO thread_files (thread_id, filename, file_type)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (thread_id, filename) DO NOTHING;
                        """, (thread_id, filename, 'docx'))
                        conn.commit()
            except Exception as db_err:
                logger.error(f"[RAG] DB register for docx failed: {db_err}", exc_info=True)
                    
        return chunks_added
    finally:
        # 5. Clean up the temporary converted PDF from /tmp
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception as cleanup_err:
                print(f"[Ingestion] Error cleaning up temporary PDF {pdf_path}: {cleanup_err}")
