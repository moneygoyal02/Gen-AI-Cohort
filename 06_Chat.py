from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

## Chain-of-Thought Prompting
# 1. Chain-of-Thought (CoT) Prompting: The model is encouraged to break down reasoning step by step before arriving at an answer.


system_prompt = """"
you are a only a math tutor. solve problems step by step before giving the final answer.

Example1:
Input: what is 24 divided by 6?
Output: First, divide 24 by 6. That gives 4. So, the answer is 4.


Example2:

Input: why sky is blue?
Output: I am math tutor. So, i cannot answer this.



"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        # {"role": "user", "content": "what is 2 + 2*0 + 2/2 - 4"} # So the final answer is -1.
        {"role": "user", "content":"what is the meaning of tokenization"} # I am math tutor. So, i cannot answer this.
    ]
    
)

print(response.choices[0].message.content)