import fitz
import sys

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    print(f"Total Pages: {len(doc)}")
    for i, page in enumerate(doc):
        print(f"\n--- PAGE {i+1} ---")
        print(page.get_text())

if __name__ == "__main__":
    pdf_path = "Assignment_Financial_Chatbot.pdf"
    extract_pdf_text(pdf_path)
