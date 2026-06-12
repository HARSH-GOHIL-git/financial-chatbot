import os
import sys

# Ensure chatbot-12 directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.rag_pipeline import expand_query

def test_query_expansion():
    print("Testing expand_query...")
    original_query = "What is the revenue growth rate of Apple in 2022?"
    queries = expand_query(original_query)
    print(f"Original Query: {original_query}")
    print(f"Resulting Queries: {queries}")
    assert isinstance(queries, list), "Should return a list"
    assert len(queries) >= 1, "Should have at least the original query"
    print("Test passed successfully!")

if __name__ == "__main__":
    test_query_expansion()
