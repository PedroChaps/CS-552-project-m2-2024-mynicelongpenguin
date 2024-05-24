"""
Given our trained models and the prompts of the # TODO P file, we make inference with all the models we have and save the outputs
"""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset

def dprint(*args, **kwargs):
    if False:
        print("[DEBUG make_inference.py]", *args, **kwargs)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Open the file with the prompts that will be passed to each model
# TODO P change the path to the prompts file
prompts = load_dataset("json", data_files="data/dpo_formatted_epfl_preference_data.jsonl", split="train")
prompts = list(map(lambda x: x["prompt"], prompts))

formatted_prompts = [
    [
        {
            "role": "system",
            "content": "You are a helpful EPFL chatbot", 
        }, 
        {
            "role": "user",
            "content": f"{prompt}", 
        },
    ] for prompt in prompts
]






# For every model we have, we open it and make inference with the prompts.
# We use the `pipeline` functions to facilitate work


model_names = [
    "rhysjones/phi-2-orange-v2",
]


for model_name in model_names:
    
    print("\x1b[2;30;42m" + f"Making inference with model {model_name}{bcolors.ENDC}" + "\x1b[0m")
    
    # TODO P which type to use? "question-answering" doesn't work
    # TODO P using a pipeline might be a problem because we need to somehow associate the "chat template" to the model (or I can just pass the tokenizer)
    pipe = pipeline("conversational", model_name, device=device)   
    
    # ----------------------- Try 1: Naive, iterating one by one -----------------------------------
    # for prompt in tqdm(prompts):
        
    #     # Prepares the message to be sent
    #     message = [
    #         {
    #             "role": "system",
    #             "content": "You are a helpful EPFL chatbot", 
    #         }, 
    #         {
    #             "role": "user",
    #             "content": f"{prompt}", 
    #         },
    #     ]
        
    #     dprint(f"{message = }")
        
    #     res = pipe(message, max_new_tokens=1024)[2]
    #     dprint(f"{res = }")
    # ----------------------------------------------------------------------------------------------
        
            
    # ----------------------- Try 2: batching ------------------------------------------------------
    # # TODO P Extremely unneficient, taking 40-60h for 25k rows. There was a suggestion of transforming data into a Dataset (uncomment for loop above to see), that might solve it.
    # batch_size = 16
    # for i in tqdm(range(0, len(formatted_prompts), batch_size)):
    #     batch_prompts = formatted_prompts[i:i + batch_size]
    #     results = pipe(batch_prompts, max_new_tokens=1024)
    #     for result in results:
    #         print(result)
    #         print("\n\n\n\n")
    # ----------------------------------------------------------------------------------------------
        
        
    # ----------------------- Try 3: Don't use pipeline---------------------------------------------
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # # TODO P should I pass add_generation_prompt=True as well?
    # tokenized_msgs = tokenizer.apply_chat_template(formatted_prompts, tokenize=True, add_generation_prompt=True, padding=True, return_tensors="pt")
    
    # outputs = model.generate(tokenized_msgs, max_new_tokens=1024)
    # ----------------------------------------------------------------------------------------------
    
    
    # ----------------------- Try 4: Make the messages a dataset -----------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_msgs = tokenizer.apply_chat_template(formatted_prompts, padding=True)
    
    # data_dict = {key: val.tolist() for key, val in tokenized_msgs.items()}
    dataset = Dataset.from_list(tokenized_msgs)
    # Error message here /\
    print(dataset)
    print(dataset[0])
    ...
    # ----------------------------------------------------------------------------------------------
    
        
    
    print("\x1b[2;30;42m" + f"Inference with model {model_name} has finished{bcolors.ENDC}")
        