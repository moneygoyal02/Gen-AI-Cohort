from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Multi modal Prompting (Text + Image)

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "What is shown in this chart?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}}
        ]}
    ]
)

print(response.choices[0].message.content)
