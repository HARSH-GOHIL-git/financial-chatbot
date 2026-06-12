import fitz

def main():
    doc = fitz.open("NeurIPS-2020-language-models-are-few-shot-learners-Paper.pdf")
    for i, page in enumerate(doc):
        text = page.get_text()
        if "Table 2.1" in text:
            print(f"Table 2.1 found on Page {i+1}")
        if "Table 2.2" in text:
            print(f"Table 2.2 found on Page {i+1}")

if __name__ == "__main__":
    main()
