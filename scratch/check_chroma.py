import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.rag_pipeline import get_vectorstore
import psycopg
from app.core.config import DB_URI

def check_db():
    print("--- ChromaDB Content ---")
    try:
        vs = get_vectorstore()
        all_data = vs.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        print(f"Total chunks in ChromaDB: {len(metadatas)}")
        
        file_counts = {}
        thread_counts = {}
        chunk_types = {}
        for m in metadatas:
            if not m:
                continue
            fn = m.get("filename", "unknown")
            tid = m.get("thread_id", "unknown")
            ct = m.get("chunk_type", "unknown")
            file_counts[fn] = file_counts.get(fn, 0) + 1
            thread_counts[tid] = thread_counts.get(tid, 0) + 1
            chunk_types[ct] = chunk_types.get(ct, 0) + 1
            
        print("By Filename:")
        for fn, count in file_counts.items():
            print(f"  - {fn}: {count} chunks")
        print("By Thread ID:")
        for tid, count in thread_counts.items():
            print(f"  - {tid}: {count} chunks")
        print("By Chunk Type:")
        for ct, count in chunk_types.items():
            print(f"  - {ct}: {count} chunks")
            
    except Exception as e:
        print(f"ChromaDB check failed: {e}")
        
    print("\n--- PostgreSQL thread_files Content ---")
    if DB_URI:
        try:
            db_uri_clean = DB_URI.strip('"').strip("'")
            with psycopg.connect(db_uri_clean) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT thread_id, filename, file_type FROM thread_files")
                    rows = cur.fetchall()
                    print(f"Total records in thread_files: {len(rows)}")
                    for r in rows:
                        print(f"  - Thread: {r[0]} | File: {r[1]} | Type: {r[2]}")
        except Exception as e:
            print(f"Postgres check failed: {e}")

if __name__ == "__main__":
    check_db()
