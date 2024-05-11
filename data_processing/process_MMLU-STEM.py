from datasets import load_dataset, load_from_disk
import gpt_wrapper
from gpt_wrapper.chat import Chat
import tqdm
import random
import os

gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = "175accd9-4b3d-42a7-998e-85600fa3d1f4"


def format_dataset():
    
    dataset = load_dataset("TIGER-Lab/MMLU-STEM")
    dataset = dataset["train"]
    
    dataset = dataset.rename_column("subject", "topic")
    dataset = dataset.rename_column("answer", "correct_choice")
    
    dataset = dataset.map(lambda x: {"correct_choice": chr(ord("A") + x["correct_choice"])})
    
    dataset = dataset.map(lambda x: {"topic": "computer_science" if x["topic"] == "high_school_computer_science" else x["topic"]})
    
    dataset = dataset.map(lambda x: {"answer": ""})    
    
    dataset.save_to_disk("data/MMLU-STEM_formatted")


def create_explanation_of_answers():

    dataset = load_from_disk("data/MMLU-STEM_formatted")
    
    template = '''
    Imagine you are a Computer Science student. I will provide you a question, some options and the correct option. You must always consider the given correct option as the only only only one correct, this is, you cannot disagree. Your job is to provide an explanation of why the provided correct option is the only correct.

    Question: {question}

    Options: {options}
    
    Correct option: {correct_option}

    Justification: Let's think step by step.
    '''

    def justify(entry):
        chat = Chat.create(f"MMLU-STEM_{random.randint(0, 999999999999)}")
        prompt = template.format(question=entry["question"], options=entry["choices"], correct_option=entry["correct_choice"])
        message = chat.ask(content=prompt)
        
        entry["answer"] = str(message)
        
        return entry

    #print(justify(dataset[3]))

    dataset_justified = dataset.map(lambda x: justify(x), num_proc=32)
    dataset_justified.save_to_disk("data/MMLU-STEM_processed")



def main():
    #format_dataset()
    create_explanation_of_answers()
    
    
if __name__ == "__main__":
    main()