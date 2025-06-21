from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# Self-Consistency Prompting
# 1. The model generates multiple responses and selects the most consistent or common answer.


# Generate multiple completions (n=3)
responses = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Which comes first egg or hen? Show your reasoning."}],
    n=3,
    temperature=0.8
)

answers = [choice.message.content for choice in responses.choices]
print(answers)