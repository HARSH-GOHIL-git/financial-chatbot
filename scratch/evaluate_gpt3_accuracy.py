import os
import sys
import json
from google import genai
from dotenv import load_dotenv

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from app.services.rag_pipeline import _search_knowledge_base_logic, get_vectorstore

# Thread ID containing 'NeurIPS-2020-language-models-are-few-shot-learners-Paper.pdf'
THREAD_ID = "962ff8e2-e3d7-411e-8a08-1fc248e299a5"

EVAL_TEST_CASES = [
    {
        "question": "How many parameters does the largest GPT-3 model have?",
        "ground_truth": "The largest GPT-3 model has 175 billion parameters (175B)."
    },
    {
        "question": "How many layers (transformer blocks) are in the 175B parameter GPT-3 model?",
        "ground_truth": "The 175B parameter model (GPT-3) has 96 layers."
    },
    {
        "question": "What dataset size was used to train GPT-3 in terms of tokens?",
        "ground_truth": "GPT-3 was trained on datasets containing a total of 300 billion tokens."
    },
    {
        "question": "What type of attention pattern is used in the GPT-3 model layers?",
        "ground_truth": "GPT-3 uses alternating dense and locally banded sparse attention patterns in its layers."
    },
    {
        "question": "What are the three context learning settings evaluated in the paper?",
        "ground_truth": "The three context learning settings evaluated are few-shot, one-shot, and zero-shot."
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
    print("Initializing GPT-3 evaluation...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[Error] GOOGLE_API_KEY not set.")
        return
        
    client = genai.Client(api_key=api_key)
    
    results = []
    total_score = 0.0
    
    for i, case in enumerate(EVAL_TEST_CASES):
        q = case["question"]
        gt = case["ground_truth"]
        print(f"\n[{i+1}/{len(EVAL_TEST_CASES)}] Question: {q}")
        
        # 1. Retrieve context using RAG retrieval pipeline
        print("Retrieving context from database...")
        context = _search_knowledge_base_logic(query=q, thread_id=THREAD_ID, text_only=True)
        
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
    report_path = "GPT3_RAG_ACCURACY_REPORT.md"
    print(f"Writing report to {report_path}...")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# GPT-3 Paper RAG Accuracy Report\n\n")
        f.write(f"This report evaluates the accuracy of the local RAG chatbot on factual questions about the **GPT-3 paper** (`NeurIPS-2020-language-models-are-few-shot-learners-Paper.pdf`).\n\n")
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
