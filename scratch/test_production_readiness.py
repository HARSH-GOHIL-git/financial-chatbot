import os
import sys
import time
import uuid
import threading
import requests

# Ensure base directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_URL = "http://localhost:8000"

def run_tests():
    print("=== STARTING LIVE BACKEND INTEGRATION TESTS ===")

    # 1. Test GET /languages
    print("\n[Test 1] Testing /languages endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/languages", timeout=60)
        assert response.status_code == 200, f"Status code is {response.status_code}"
        data = response.json()
        assert "languages" in data, "Response body should contain 'languages'"
        print(f"✓ /languages successfully fetched {len(data['languages'])} languages.")
    except Exception as e:
        print(f"✗ /languages test failed: {e}")
        return False

    # 2. Test PDF Upload & Ingestion
    print("\n[Test 2] Testing /upload-pdf endpoint...")
    test_thread_id = f"test_thread_{uuid.uuid4().hex[:8]}"
    pdf_file_path = os.path.abspath("order.pdf")
    
    if not os.path.exists(pdf_file_path):
        print(f"Warning: 'order.pdf' not found. Looking for other pdfs...")
        for f in os.listdir("."):
            if f.endswith(".pdf"):
                pdf_file_path = os.path.abspath(f)
                break

    print(f"Uploading file: {pdf_file_path} for thread_id: {test_thread_id}")
    try:
        with open(pdf_file_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_file_path), f, "application/pdf")}
            params = {"thread_id": test_thread_id}
            response = requests.post(f"{BASE_URL}/upload-pdf", files=files, params=params, timeout=120)
            
        assert response.status_code == 200, f"Status code is {response.status_code}"
        data = response.json()
        assert "status" in data and "message" in data, "Response should return status and message"
        print(f"✓ /upload-pdf successful! Message: {data['message']}")
    except Exception as e:
        print(f"✗ /upload-pdf test failed: {e}")
        return False

    # 3. Test /chat Query (Sync RAG)
    print("\n[Test 3] Testing /chat (RAG retrieval and response)...")
    try:
        payload = {
            "message": "What is the content of the uploaded PDF?",
            "thread_id": test_thread_id,
            "language": "English"
        }
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60)
        assert response.status_code == 200, f"Status code is {response.status_code}"
        data = response.json()
        assert "reply" in data, "Response should contain reply"
        print("✓ /chat successful! Reply preview:")
        print(f"--- Reply: {data['reply'][:150]}...")
    except Exception as e:
        print(f"✗ /chat sync test failed: {e}")
        return False

    # 4. Test /chat_stream (Streaming token output)
    print("\n[Test 4] Testing /chat_stream (Streaming token output)...")
    try:
        payload = {
            "message": "Explain the key topics or items listed in this document in detail.",
            "thread_id": test_thread_id,
            "language": "English"
        }
        
        response = requests.post(f"{BASE_URL}/chat_stream", json=payload, stream=True, timeout=60)
        assert response.status_code == 200, f"Status code is {response.status_code}"
        
        chunk_count = 0
        text_received = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                text_received += decoded_line + "\n"
                chunk_count += 1
                if chunk_count > 10:
                    break
        print(f"✓ /chat_stream successful! Received {chunk_count} streaming lines.")
    except Exception as e:
        print(f"✗ /chat_stream test failed: {e}")
        return False

    # 5. Test Thread Cancellation via /stop
    print("\n[Test 5] Testing /stop cancellation endpoint...")
    try:
        cancel_thread_id = f"cancel_thread_{uuid.uuid4().hex[:8]}"
        
        # Start a stream in a background thread
        def stream_worker():
            payload = {
                "message": "Explain quantum computing in detail, step by step, list 20 benefits.",
                "thread_id": cancel_thread_id,
                "language": "English"
            }
            try:
                requests.post(f"{BASE_URL}/chat_stream", json=payload, timeout=60)
            except Exception:
                pass
                
        t = threading.Thread(target=stream_worker)
        t.start()
        
        # Wait a moment for generation to start
        time.sleep(2.0)
        
        # Send cancel request
        response = requests.post(f"{BASE_URL}/stop", json={"thread_id": cancel_thread_id}, timeout=10)
        assert response.status_code == 200, f"Stop request failed with {response.status_code}"
        data = response.json()
        print(f"✓ /stop response: {data}")
        t.join(timeout=10)
    except Exception as e:
        print(f"✗ /stop cancellation test failed: {e}")
        return False

    # 6. Test Multi-Threaded Concurrent Load (Race Conditions & Thread Safety)
    print("\n[Test 6] Testing concurrent requests for thread safety...")
    errors = []
    
    def worker_concurrency(worker_id):
        worker_thread_id = f"worker_{worker_id}_{uuid.uuid4().hex[:4]}"
        try:
            # First upload a document
            with open(pdf_file_path, "rb") as f:
                files = {"file": (os.path.basename(pdf_file_path), f, "application/pdf")}
                params = {"thread_id": worker_thread_id}
                requests.post(f"{BASE_URL}/upload-pdf", files=files, params=params, timeout=60)
                
            # Then perform query
            payload = {
                "message": "What is the order or content in the document?",
                "thread_id": worker_thread_id,
                "language": "English"
            }
            res = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60)
            if res.status_code != 200:
                errors.append(f"Worker {worker_id} request failed with status {res.status_code}. Response: {res.text}")
        except Exception as err:
            errors.append(f"Worker {worker_id} exception: {err}")

    threads = []
    for idx in range(3):
        t = threading.Thread(target=worker_concurrency, args=(idx,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if errors:
        print(f"✗ Concurrent request test failed with errors: {errors}")
        return False
    else:
        print("✓ Concurrent requests completed with NO race conditions or crashes!")

    print("\n=== ALL INTEGRATION TESTS PASSED SUCCESSFULLY! ===")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
