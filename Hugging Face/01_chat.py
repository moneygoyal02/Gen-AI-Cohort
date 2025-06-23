import os
from dotenv import load_dotenv


load_dotenv()  # loads .env variables into os.environ


model_name = "google/gemma-3-1b-it" # specify the model name
 
from transformers import AutoTokenizer  # import the AutoTokenizer class from transformers library

tokenizer = AutoTokenizer.from_pretrained(model_name)   # load the tokenizer for the specified model

print(tokenizer("hey, how are you? "))  # print the tokenizer object to verify it has been loaded correctly  
# {'input_ids': [2, 36935, 236764, 1217, 659, 611, 236881, 236743], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
# print(tokenizer.get_vocab())
# print(tokenizer.get_vocab_size())

input_tokens = tokenizer("hey, how are you?")["input_ids"]
print(input_tokens)

# importing the model
from transformers import AutoModelForCausalLM # import the AutoModelForCausalLM class from transformers library, automodalforcasuallm is used for causal language modeling tasks

import torch
torch.set_float32_matmul_precision('high')
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)  # load the model for the specified model

from transformers import pipeline

gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)



print(gen_pipeline("Hey there", max_new_tokens=25))