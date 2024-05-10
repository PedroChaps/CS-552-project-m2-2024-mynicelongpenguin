from datasets import load_dataset, load_from_disk
import gpt_wrapper
from gpt_wrapper.chat import Chat
import tqdm
import random
import os

dataset = load_from_disk("../data/Open-Platypus_formatted")

# Set up GPT Wrapper
gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = "7a1ed46e-6371-445e-a383-cc6ccf0525fd"

# Template for the prompt
template = '''
Imagine you are a Math/Physics/Computer Science student. 
I will provide you a question and its corresponding correct answer. 
You must always consider the given answer as correct, this is, you cannot disagree. 
Your job is to provide the justification to why that answer is correct.

Question: {question}

Correct answer: {answer}

Justification: Let's think step by step.
'''

# Function to justify the answer
def justify(entry):
    # Check if the answer is shorter than 10 words
    if len(entry["answer"].split()) < 10:
        # Create a GPT chat instance
        chat = Chat.create(f"openplatypus_{random.randint(0, 999999999999)}")
        # Construct the prompt
        prompt = template.format(question=entry["question"], answer=entry["answer"])
        # Get GPT's response
        message = chat.ask(content=prompt)
        # Update the answer with the modified justification
        entry["answer"] = str(message)
    return entry

# Justify the dataset
dataset_justified = dataset.map(justify, num_proc=32)

# Save the justified dataset
dataset_justified.save_to_disk("../data/open_platypus_justified")