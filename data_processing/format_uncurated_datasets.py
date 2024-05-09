from datasets import load_dataset, concatenate_datasets
import random

dataset = load_dataset("garage-bAInd/Open-Platypus", split="train")

def format_entry(entry, topic):
    if topic == "random":
        topic = random.choice(["math", "physics", "code"])
    return {
        "question": entry["instruction"],
        "choices" : None,
        "correct_choice": None,
        "topic": topic,
        "answer": entry["output"]
    }

dataset = dataset.filter(lambda x: x['data_source'] in ['MATH/PRM-800K', 'airoboros', 'scienceqa', 'leetcode_ne', 'ARB', 'scibench', 'tigerbot-kaggle'])

# separate datasets by topics
dataset_math = dataset.filter(lambda x: x['data_source'] in ['MATH/PRM-800K', 'ARB'])
dataset_code = dataset.filter(lambda x: x['data_source'] in ['leetcode_ne', 'tigerbot-kaggle'])
dataset_physics = dataset.filter(lambda x: x['data_source'] in ['scienceqa', 'scibench'])
dataset_random = dataset.filter(lambda x: x['data_source'] in ['airoboros'])

# format entries
dataset_math = dataset_math.map(lambda x: format_entry(x, "math"), num_proc=8)
dataset_code = dataset_code.map(lambda x: format_entry(x, "code"), num_proc=8)
dataset_physics = dataset_physics.map(lambda x: format_entry(x, "physics"), num_proc=8)
dataset_random = dataset_random.map(lambda x: format_entry(x, "random"), num_proc=8)


formatted_dataset = concatenate_datasets([dataset_math, dataset_code, dataset_physics, dataset_random])

# formatted_dataset = dataset.map(format_entry, num_proc=8)

formatted_dataset.save_to_disk("../data/Open-Platypus_formatted")