import os
import sys
from dotenv import load_dotenv

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from app.services.rag_pipeline import get_vectorstore

def find_thread():
    vs = get_vectorstore()
    all_data = vs.get(include=["metadatas"])
    metadatas = all_data.get("metadatas", [])
    
    found = {}
    for m in metadatas:
        if not m:
            continue
        fn = m.get("filename", "unknown")
        tid = m.get("thread_id", "unknown")
        if "NeurIPS" in fn or "few-shot" in fn or "few_shot" in fn:
            found[tid] = fn
            
    print("Found GPT-3 matches:")
    for tid, fn in found.items():
        print(f"Thread ID: {tid} | File: {fn}")

if __name__ == "__main__":
    find_thread()
