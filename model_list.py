from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
api = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api)

for model in client.models.list():
    print(model.name)