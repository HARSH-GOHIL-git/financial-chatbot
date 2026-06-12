import os
import sys
import psycopg
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from app.services.rag_pipeline import get_vectorstore
from app.main import DB_URI, chatbot

def main():
    if not DB_URI:
        print("DB_URI not set")
        return
        
    db_uri_clean = DB_URI.strip('"').strip("'")
    
    # Let's inspect the threads in the database
    with psycopg.connect(db_uri_clean) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT thread_id FROM message_timestamps")
            threads = cur.fetchall()
            print("Threads in message_timestamps:", [t[0] for t in threads])
            
            # Let's pick a thread and inspect message IDs
            if threads:
                tid = threads[0][0]
                print(f"\nInspecting thread: {tid}")
                cur.execute("SELECT message_id, timestamp FROM message_timestamps WHERE thread_id = %s", (tid,))
                ts_rows = cur.fetchall()
                print("Timestamps in DB:")
                for mid, ts in ts_rows:
                    print(f"  ID: {mid} | Timestamp: {ts}")
                
                # Fetch LangGraph state messages
                import asyncio
                loop = asyncio.get_event_loop()
                state = loop.run_until_complete(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: chatbot.get_state({"configurable": {"thread_id": tid}})
                    )
                )
                if state:
                    messages = state.values.get("messages", [])
                    print("\nMessages in LangGraph state:")
                    for msg in messages:
                        mid = getattr(msg, "id", None)
                        print(f"  Class: {msg.__class__.__name__} | ID: {mid} | Content: {str(msg.content)[:40]}")
                else:
                    print("No state found for thread")

if __name__ == "__main__":
    main()
