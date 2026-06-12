import fitz

def main():
    doc = fitz.open("NeurIPS-2020-language-models-are-few-shot-learners-Paper.pdf")
    for i, page in enumerate(doc):
        text = page.get_text()
        if "125M" in text or "125 million" in text:
            if "96" in text or "layers" in text.lower() or "Table" in text:
                print(f"Page {i+1} might contain the model architecture table. Excerpt:")
                lines = text.split("\n")
                for line in lines[:15]:
                    print("  ", line)
                print("-" * 50)

if __name__ == "__main__":
    main()
