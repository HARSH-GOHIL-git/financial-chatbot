import fitz

def main():
    doc = fitz.open("NeurIPS-2020-language-models-are-few-shot-learners-Paper.pdf")
    for i, page in enumerate(doc):
        text = page.get_text()
        if "Table A.1" in text or "table a.1" in text.lower():
            print(f"Table A.1 found on Page {i+1}. Text:")
            print(text[:1000])
            print("-" * 50)

if __name__ == "__main__":
    main()
