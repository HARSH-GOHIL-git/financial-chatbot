import os
import sys
import shutil
from dotenv import load_dotenv

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from app.services.rag_pipeline import index_pdf_file
from app.core.config import DB_URI

BERT_PATH = "Bert paper.pdf"
GPT3_PATH = "NeurIPS-2020-language-models-are-few-shot-learners-Paper.pdf"

BERT_THREAD = "d8125cd0-59c3-4175-baf8-881c5c209294"
GPT3_THREAD = "962ff8e2-e3d7-411e-8a08-1fc248e299a5"

def main():
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        print(f"Deleting old ChromaDB at '{db_path}'...")
        shutil.rmtree(db_path)
    
    print("\n--- Indexing BERT paper ---")
    if os.path.exists(BERT_PATH):
        chunks_added = index_pdf_file(
            temp_path=BERT_PATH,
            filename=os.path.basename(BERT_PATH),
            thread_id=BERT_THREAD,
            db_uri=DB_URI,
            smolvlm=None  # Bypassing local SmolVLM for indexing text
        )
        print(f"Successfully indexed BERT paper. Added {chunks_added} chunks.")
    else:
        print(f"[Error] BERT paper not found at '{BERT_PATH}'")

    print("\n--- Indexing GPT-3 paper ---")
    if os.path.exists(GPT3_PATH):
        chunks_added = index_pdf_file(
            temp_path=GPT3_PATH,
            filename=os.path.basename(GPT3_PATH),
            thread_id=GPT3_THREAD,
            db_uri=DB_URI,
            smolvlm=None
        )
        print(f"Successfully indexed GPT-3 paper. Added {chunks_added} chunks.")
    else:
        print(f"[Error] GPT-3 paper not found at '{GPT3_PATH}'")

    print("\nRe-indexing completed successfully!")

if __name__ == "__main__":
    main()
