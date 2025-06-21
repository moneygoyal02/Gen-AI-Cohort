from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Instruction Prompting

system_prompt = """

Answer the question in 2 Sentences and include a fact related to it.

"""

result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role":"user", "content":"what is the capital of India?"}
    ]
)

print(result.choices[0].message.content)