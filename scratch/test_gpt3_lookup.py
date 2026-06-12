import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from app.services.rag_pipeline import get_vectorstore, search_knowledge_base

def main():
    vs = get_vectorstore()
    # Let's search for "Table 2.1" or "layers" in the GPT-3 thread
    thread_id = "962ff8e2-e3d7-411e-8a08-1fc248e299a5"
    
    # Query Chroma directly for all chunks on page 7 or 8 (0-indexed page 7 is page 8)
    res = vs.get(where={
        "$and": [
            {"thread_id": thread_id},
            {"page": 7}
        ]
    })
    
    print("=== Chunks on Page 8 ===")
    for text, meta in zip(res.get("documents", []), res.get("metadatas", [])):
        print(f"Type: {meta.get('chunk_type')} | Length: {len(text)}")
        print(text[:300])
        print("-" * 50)
        
    print("\n=== Similarity Search for '96 layers' ===")
    docs = vs.similarity_search("96 layers or transformer blocks in the 175B model", k=10, filter={"thread_id": thread_id})
    for i, doc in enumerate(docs):
        print(f"Rank {i+1} | Page {doc.metadata.get('page')+1} | Type {doc.metadata.get('chunk_type')}")
        print(doc.page_content[:200])
        print("-" * 50)

if __name__ == "__main__":
    main()
