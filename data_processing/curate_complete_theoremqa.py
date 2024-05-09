from datasets import load_dataset
import gpt_wrapper
from gpt_wrapper.chat import Chat
import tqdm
import random
import os

# CHECK THIS IS CORRECT BEFORE RUNNING, USED TO SAVE THE DATASET
os.chdir(os.path.dirname(os.path.realpath(__file__)))

gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = "4eadeb76-03d6-40cc-b858-27b0da441ac5"

dataset = load_dataset("TIGER-Lab/TheoremQA", split="test")


dataset_no_pic = dataset.filter(lambda x: x["Picture"] is None)

dataset_no_pic = dataset_no_pic.remove_columns("Picture")

################################## EXTEND DATASET WITH JUSTIFICATIONS ##################################
template = '''
Imagine you are a Math and Physics student. I will provide you a question and its corresponding correct answer. You must always consider the given answer as correct, this is, you cannot disagree. Your job is to provide the justification to why that answer is correct.

Question: {question}


Correct answer: {answer}


Justification: Let's think step by step.
'''

def justify(entry):
    chat = Chat.create(f"theoremqa_{random.randint(0, 999999999999)}")
    prompt = template.format(question=entry["Question"], answer=entry["Answer"])
    message = chat.ask(content=prompt)
    return {"Question": entry["Question"], "Answer": entry["Answer"], "Justification": str(message)}

dataset_justified = dataset_no_pic.map(lambda x: justify(x), num_proc=32)
dataset_justified.save_to_disk("../data/theoremqa_justified_2")