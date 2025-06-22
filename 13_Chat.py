from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


#Role-Playing Prompting

system_prompt = """
You are a doctor. Respond to medical queries in a professional tone.
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "I have a sore throat and mild fever. What should I do?"}
    ]
)

print(response.choices[0].message.content)
