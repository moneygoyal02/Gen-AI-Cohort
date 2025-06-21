from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()   

# few short prompting
# 1. Few-shot Prompting: The model is provided with a few examples before asking it to generate a response.


system_prompt = """
you are a language model that translates English to French.

Example 1:
Input: Hello
Output: Bonjour

Example 2:
Input: Thank you
Output: Merci

Now translate the following English text to French:
"""

result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        { "role": "system", "content": system_prompt},
        { "role": "user", "content": "Good Night" } # few short prompting
    ]
)

print(result.choices[0].message.content)