import fitz

def main():
    doc = fitz.open("NeurIPS-2020-language-models-are-few-shot-learners-Paper.pdf")
    page = doc[7] # Page 8 is index 7
    
    print("=== RAW TEXT OF PAGE 8 ===")
    print(page.get_text())
    
    print("\n=== DETECTED TABLES ON PAGE 8 ===")
    tables_finder = page.find_tables()
    tables = tables_finder.tables if tables_finder else []
    print(f"Number of tables found: {len(tables)}")
    for i, t in enumerate(tables):
        print(f"\nTable {i+1} bbox: {t.bbox}")
        print("Markdown:")
        print(t.to_markdown())

if __name__ == "__main__":
    main()
