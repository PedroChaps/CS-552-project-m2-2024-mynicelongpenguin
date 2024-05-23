"""
Given our trained models and the prompts of the # TODO file, we make inference with all the models we have and save the outputs
"""
import json
from transformers import pipeline

def read_jsonl(path, mode="r", **kwargs):
    ls = []
    with open(path, mode, **kwargs) as f:
        for line in f:
            ls.append(json.loads(line))
    return ls


# Open the file with the prompts that will be passed to each model
# TODO change the path to the prompts file
prompts = read_jsonl("data/dpo_formatted_epfl_preference_data.jsonl")
prompts = list(map(lambda x: x["prompt"], prompts))

# For every model we have, we open it and make inference with the prompts.
# We use the `pipeline` functions to facilitate work

pipe = pipeline("question-answering", "rhysjones/phi-2-orange-v2")
