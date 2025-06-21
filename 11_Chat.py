from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Direct Answer Prompting


system_prompt = """
Answer only with the direct answer. No explanation.
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is 8 squared?"}
    ]
)

print(response.choices[0].message.content)