import fitz

def main():
    doc = fitz.open("NeurIPS-2020-language-models-are-few-shot-learners-Paper.pdf")
    for i, page in enumerate(doc):
        text = page.get_text()
        if "96" in text:
            print(f"Page {i+1} contains '96'. Excerpts:")
            for line in text.split("\n"):
                if "96" in line:
                    print("  ", line)

if __name__ == "__main__":
    main()
