from transformers import (
    # AutoModelForCausalLM,
    AutoTokenizer,

)

from datasets import load_from_disk

# template = '''
# ### Question:
# {question}

# ### Answer:
# {answer}
# '''

# template_MCQ = '''
# ### Question:
# {question}

# ### Choices:
# A: {choices[0]}
# B: {choices[1]}
# C: {choices[2]}
# D: {choices[3]}

# ### Correct Choice:
# {chr(ord('A') + correct_choice)}

# ### Explanation:
# {answer}
# '''

tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True,
        )
tokenizer.pad_token = tokenizer.eos_token


dataset = load_from_disk("data/MMLU-STEM_final_fixedCorrectChoice")


def collate_row(row):
    
    question = row["question"]
    answer = row["answer"]
    choices = row["choices"]
    correct_choice = row["correct_choice"]
    
    if choices in [None, "", "none", "None", "null", "Null"]:
        prompt = f'''\
            ### Question:
            {question}


            ### Answer:
            {answer}
        '''
    
    
    else:
        prompt = f'''\
            ### Question:
            {question}


            ### Choices:
            {f"{chr(10)}".join(f"{chr(ord('A') + i)}: {choice}" for i, choice in enumerate(choices))}


            ### Explanation:
            {answer}
            
            
            ### Correct Choice:
            {chr(ord('A') + correct_choice)}
        '''
    
    # Removes the leading whitespaces because of the tabs in the triple quoted strings
    prompt = '\n'.join([m.lstrip() for m in prompt.split('\n')])
    
    return {"text": prompt}
    
    
if __name__ == "__main__":
    new_dataset = dataset.map(collate_row)
    print(new_dataset)
    