# import subprocess

# try:
#     gpu_info = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
#     print(gpu_info)
# except Exception as e:
#     print('Not connected to a GPU')


import os
from dotenv import load_dotenv


load_dotenv() 

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "google/gemma-3-1b-it"


tokenizer = AutoTokenizer.from_pretrained(model_name) 

tokenizer("Hello, World!")

input_conversation = [
    { "role": "user", "content": "Which is the best place to learn GenAI?" },
    { "role": "assistant", "content": "The best place to learn AI is" }
]

input_tokens = tokenizer.apply_chat_template(
    conversation=input_conversation,
    tokenize=True,
)

print(input_tokens)


input_detokens = tokenizer.apply_chat_template(
    conversation=input_conversation,
    tokenize=False,  # gives detokenized output if it is true then it will return tokenized output
    continue_final_message=True,
)

print(input_detokens)

output_label = "GenAI Cohort 1.0 by ChaiCode and Piyush Garg use Coupon PIYUSH10 Please"
full_conversation = input_detokens + output_label + tokenizer.eos_token


print(full_conversation)

input_tokenized = tokenizer(full_conversation, return_tensors="pt", add_special_tokens=False).to(device)["input_ids"]

print(input_tokenized)


input_ids = input_tokenized[:, :-1].to(device)
target_ids = input_tokenized[:, 1:].to(device)
print(f"input_ids: {input_ids}")
print(f"target_ids: {target_ids}")

import torch.nn as nn
def calculate_loss(logits, labels):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    cross_entropy = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return cross_entropy


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation='eager'
).to(device)


from torch.optim import AdamW
model.train()

optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

for _ in range(10):
  out = model(input_ids=input_ids)
  loss = calculate_loss(out.logits, target_ids).mean()
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  print(loss.item())
  
  input_prompt = [
    { "role": "user", "content": "Which is the best place to learn GenAI?" }
]

input = tokenizer.apply_chat_template(
    conversation=input_prompt,
    return_tensors="pt",
    tokenize=True,
).to(device)

output = model.generate(input, max_new_tokens=35)
print(tokenizer.batch_decode(output, skip_special_tokens=True))