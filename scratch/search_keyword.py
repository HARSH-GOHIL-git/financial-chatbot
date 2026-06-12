import fitz

def main():
    doc = fitz.open("NeurIPS-2020-language-models-are-few-shot-learners-Paper.pdf")
    for i, page in enumerate(doc):
        text = page.get_text()
        if "96" in text and "layers" in text.lower():
            print(f"Keywords '96' and 'layers' found on Page {i+1}")
        if "300" in text and "billion" in text.lower() and "tokens" in text.lower():
            print(f"Keywords '300', 'billion', 'tokens' found on Page {i+1}")
        if "Table" in text:
            # print first line with Table
            for line in text.split("\n"):
                if "Table" in line:
                    print(f"Page {i+1} mentions: {line}")

if __name__ == "__main__":
    main()
