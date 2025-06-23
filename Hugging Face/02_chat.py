import os
from dotenv import load_dotenv


load_dotenv() 

from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "google/gemma-3-1b-it"


tokenizer = AutoTokenizer.from_pretrained(model_name) 

input_prompt = [" The Capital of India is"]

tokenized = tokenizer(input_prompt, return_tensors="pt")

print(tokenized["input_ids"]) # tensor([[    2,   669, 16930,   529,  4673,   563]])

import torch
torch.set_float32_matmul_precision('high')
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)

gen_result = model.generate(tokenized["input_ids"], max_new_tokens=25)

# tensor([[     2,    669,  16930,    529,   4673,    563,   1492,    528,    506,
#            4916,    529,    506,  14815,    529,   6863,  20004, 236764,    837,
#             563,   1208,    506,   2256,    529,    506,  13946,   7079,    529,
#            4673, 236761,    107,    106]])
# this is the predicted tokens by model

print(gen_result)

output = tokenizer.batch_decode(gen_result)
print(output)