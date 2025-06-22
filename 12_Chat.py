from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Persona-based Prompting

system_prompt = """
You are Albert Einstein. Respond in his style with wisdom and curiosity.
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is the importance of imagination?"}
    ]
)

print(response.choices[0].message.content)
