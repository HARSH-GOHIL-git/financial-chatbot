import os
import sys
import json
from google import genai
from dotenv import load_dotenv

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()


from app.services.rag_pipeline import _search_knowledge_base_logic, get_vectorstore

# Thread ID containing 'Bert paper.pdf'
THREAD_ID = "d8125cd0-59c3-4175-baf8-881c5c209294"

EVAL_TEST_CASES = [
    {
        "question": "What does BERT stand for?",
        "ground_truth": "BERT stands for Bidirectional Encoder Representations from Transformers."
    },
    {
        "question": "What are the two pre-training tasks used in BERT?",
        "ground_truth": "The two pre-training tasks are Masked Language Model (MLM) and Next Sentence Prediction (NSP)."
    },
    {
        "question": "How many layers (transformer blocks) are in BERT-Base and BERT-Large?",
        "ground_truth": "BERT-Base has 12 layers (L=12) and BERT-Large has 24 layers (L=24)."
    },
    {
        "question": "What is the size of the hidden dimension (H) in BERT-Base and BERT-Large?",
        "ground_truth": "BERT-Base has a hidden size of 768 (H=768) and BERT-Large has a hidden size of 1024 (H=1024)."
    },
    {
        "question": "What vocabulary size is used in BERT?",
        "ground_truth": "BERT uses a WordPiece vocabulary of 30,000 tokens."
    }
]

def call_evaluator_llm(question, ground_truth, response):
    """Uses Gemini to evaluate response correctness against ground truth."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return 0.0, "API key missing"
    
    client = genai.Client(api_key=api_key)
    prompt = f"""You are an objective evaluator. Compare the following generated response to the ground truth answer for the given question.
    
Question: {question}
Ground Truth: {ground_truth}
Generated Response: {response}

Judge the generated response based on:
1. Is it factually correct?
2. Does it answer the question fully based on the ground truth?

Respond with a JSON object containing:
- "score": A float between 0.0 (totally wrong/hallucination/missing) and 1.0 (completely correct and complete).
- "reason": A short 1-2 sentence explanation of your score.

Return ONLY the raw JSON object, no markdown blocks, no text outside the JSON."""

    try:
        res = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
        text = res.text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "", 1).rsplit("```", 1)[0].strip()
        elif text.startswith("```"):
            text = text.replace("```", "", 1).rsplit("```", 1)[0].strip()
        
        data = json.loads(text)
        return float(data.get("score", 0.0)), data.get("reason", "No reason provided")
    except Exception as e:
        return 0.0, f"Evaluation error: {e}"

def run_evaluation():
    print("Initializing evaluation...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[Error] GOOGLE_API_KEY not set.")
        return
        
    client = genai.Client(api_key=api_key)
    
    # Check if thread exists in vector store
    vs = get_vectorstore()
    exists = vs.get(where={"thread_id": THREAD_ID}, limit=1)
    if not exists or not exists.get("ids"):
        print(f"[Warning] Thread ID '{THREAD_ID}' not found in ChromaDB. Using first available thread instead.")
        all_data = vs.get(include=["metadatas"], limit=100)
        metadatas = all_data.get("metadatas", [])
        active_thread = None
        for m in metadatas:
            if m and m.get("thread_id"):
                active_thread = m.get("thread_id")
                break
        if not active_thread:
            print("[Error] No active threads/documents found in ChromaDB. Please index a PDF first.")
            return
        print(f"Using alternative active thread: {active_thread}")
        thread_id = active_thread
    else:
        thread_id = THREAD_ID
        print(f"Target thread '{thread_id}' is active in ChromaDB.")

    results = []
    total_score = 0.0
    
    for i, case in enumerate(EVAL_TEST_CASES):
        q = case["question"]
        gt = case["ground_truth"]
        print(f"\n[{i+1}/{len(EVAL_TEST_CASES)}] Question: {q}")
        
        # 1. Retrieve context using RAG retrieval pipeline
        print("Retrieving context from database...")
        context = _search_knowledge_base_logic(query=q, thread_id=thread_id, text_only=True)
        
        # 2. Call Gemini to answer the question using the retrieved context
        print("Generating answer from context...")
        answer_prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided context retrieved from the document.
        If the context does not contain the answer, say "I cannot find the answer in the retrieved context."
        
Context:
{context}

Question: {q}"""
        
        try:
            res = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=answer_prompt
            )
            response_text = res.text.strip()
        except Exception as e:
            response_text = f"[Error generating answer: {e}]"
            
        print(f"Generated Response: {response_text}")
        
        # 3. Grade the answer using LLM-as-a-judge
        print("Grading response...")
        score, reason = call_evaluator_llm(q, gt, response_text)
        print(f"Score: {score*100}% | Reason: {reason}")
        
        total_score += score
        results.append({
            "question": q,
            "ground_truth": gt,
            "retrieved_context": context,
            "response": response_text,
            "score": score,
            "reason": reason
        })

    avg_accuracy = (total_score / len(EVAL_TEST_CASES)) * 100.0
    print(f"\n=== Evaluation Complete ===")
    print(f"Average RAG Accuracy: {avg_accuracy:.2f}%")
    
    # Save the report
    report_path = "RAG_ACCURACY_REPORT.md"
    print(f"Writing report to {report_path}...")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Chatbot RAG Retrieval and Answering Accuracy Report\n\n")
        f.write(f"This report evaluates the accuracy of the local RAG chatbot on a series of factual questions about the **BERT paper** (which is indexed in the ChromaDB vector store).\n\n")
        f.write(f"## Overall Score\n")
        f.write(f"- **Actual Accuracy**: **{avg_accuracy:.2f}%**\n\n")
        f.write(f"## Test Case Details\n\n")
        for i, res in enumerate(results):
            f.write(f"### Test Case {i+1}: {res['question']}\n")
            f.write(f"- **Ground Truth**: {res['ground_truth']}\n")
            f.write(f"- **Generated Answer**: {res['response']}\n")
            f.write(f"- **Score**: **{res['score']*100:.1f}%**\n")
            f.write(f"- **Evaluator Reason**: {res['reason']}\n\n")
            f.write(f"<details><summary>Retrieved Context Chunks</summary>\n\n```text\n{res['retrieved_context']}\n```\n\n</details>\n\n---\n\n")
        
        f.write(f"\n*Generated automatically by evaluation script on 2026-06-12.*")
        
    print("Done!")

if __name__ == "__main__":
    run_evaluation()
