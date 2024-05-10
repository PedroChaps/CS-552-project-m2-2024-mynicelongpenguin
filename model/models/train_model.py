import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import notebook_login
from model_dpo import AutoDPOModelForCausalLM

notebook_login()

torch.set_default_device('cuda')

pretrained_model_id = "microsoft/phi-2"

pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_id, trust_remote_code=True, device_map="auto", attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, add_eos_token=True, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token

pretrained_model.gradient_checkpointing_enable()

model = AutoDPOModelForCausalLM.from_pretrained(pretrained_model_id, trust_remote_code=True, device_map="auto", attn_implementation="flash_attention_2")



















### try inference
new_prompt = """###System: 
Read the references provided and answer the corresponding question.
###References:
[1] For most people, the act of reading is a reward in itself. However, studies show that reading books also has benefits that range from a longer life to career success. If you’re looking for reasons to pick up a book, read on for seven science-backed reasons why reading is good for your health, relationships and happiness.
[2] As per a study, one of the prime benefits of reading books is slowing down mental disorders such as Alzheimer’s and Dementia  It happens since reading stimulates the brain and keeps it active, which allows it to retain its power and capacity.
[3] Another one of the benefits of reading books is that they can improve our ability to empathize with others. And empathy has many benefits – it can reduce stress, improve our relationships, and inform our moral compasses.
[4] Here are 10 benefits of reading that illustrate the importance of reading books. When you read every day you:
[5] Why is reading good for you? Reading is good for you because it improves your focus, memory, empathy, and communication skills. It can reduce stress, improve your mental health, and help you live longer. Reading also allows you to learn new things to help you succeed in your work and relationships.
###Question:
Why is reading books widely considered to be beneficial?
###Answer:
"""

inputs = tokenizer(new_prompt, return_tensors="pt", return_attention_mask=False, padding=True, truncation=True)
outputs = pretrained_model.generate(**inputs, repetition_penalty=1.0, max_length=1000)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(result)

