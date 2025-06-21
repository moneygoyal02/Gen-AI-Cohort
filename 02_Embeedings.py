from dotenv import load_dotenv
import os
from google import genai


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

text = "Eiffel Tower is in Paris and is a famous landmark, it is 324 meters tall"

# Generate embedding using the experimental 3‑k‑dim model
result = client.models.embed_content(
    model="gemini-embedding-exp-03-07",
    contents=text
)

# Access embedding vector
embedding = result.embeddings[0].values
print("Vector Embedding (len={}):".format(len(embedding)))
print(embedding[:10], "...")  # preview first 10 values
print("Full embedding:", embedding)
