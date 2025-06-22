from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Contextual Prompting

system_prompt = """
The user is writing a research paper on climate change and needs help forming strong arguments.
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What are some economic impacts of climate change?"}
    ]
)

print(response.choices[0].message.content)
