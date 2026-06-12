import fitz

def main():
    doc = fitz.open("NeurIPS-2020-language-models-are-few-shot-learners-Paper.pdf")
    print(f"Total pages: {len(doc)}")
    for i, page in enumerate(doc):
        text = page.get_text()
        if "Table A." in text:
            print(f"Page {i+1} mentions/contains 'Table A.'")
        if "Table 2.1" in text:
            print(f"Page {i+1} mentions/contains 'Table 2.1'")
            
    # Let's search for "Table" on all pages and print the surrounding lines
    for i, page in enumerate(doc):
        text = page.get_text()
        if "Table" in text:
            print(f"\n--- Page {i+1} Tables ---")
            for line in text.split("\n"):
                if "Table" in line:
                    print("  ", line)

if __name__ == "__main__":
    main()
