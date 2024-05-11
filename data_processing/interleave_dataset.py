# %%
from datasets import load_dataset, load_from_disk, interleave_datasets
import tqdm
import random
import os

platypus = load_from_disk("../data/open_platypus_justified_2")
physics_arxiv = load_from_disk("../data/arxiv-physics-instruct-tune-30k-formatted")
theoremqa = load_from_disk("../data/theoremqa_justified_formatted_2")
mmlu = load_from_disk("../data/MMLU-STEM_processed")


# Access the specific split for dataset2
physics_arxiv = physics_arxiv['train']

# # %%
# platypus[0]

# # %%
# dataset2_train[0]

# # %%
# theoremqa[0]

# # %%
datasets_list = [platypus, physics_arxiv, theoremqa, mmlu]

# %%
dataset_probabilities = [0.5, 0.3, 0.2]

interleaved_dataset = interleave_datasets(
    datasets=datasets_list,
    probabilities=None,  # or use probabilities list defined above
    seed=42,           # optional: use a seed for reproducibility
    stopping_strategy='first_exhausted'  # or 'all_exhausted'
)

# If we need only 30k rows 
if len(interleaved_dataset) > 30000:
    interleaved_dataset = interleaved_dataset.select(range(30000))

interleaved_dataset.save_to_disk("../data/interleaved_dataset")

# %%
interleaved_dataset = load_from_disk("../data/interleaved_dataset")
interleaved_dataset[0]

# %% [markdown]
# ### If we want to test different dataset probability splits

# %%
dataset_probabilities = [
    [0.33, 0.33, 0.34],
    [0.50, 0.30, 0.20],
    [0.20, 0.50, 0.30],
    [0.10, 0.45, 0.45],
    [0.25, 0.25, 0.50]
]

# Base path for saving datasets
base_save_path = "../data/interleaved_variations/"

for i, probs in enumerate(dataset_probabilities):
    # Interleave datasets with current probability distribution
    interleaved_dataset = interleave_datasets(
        datasets=datasets_list,
        probabilities=probs,
        seed=42,  # Ensuring reproducibility
        stopping_strategy='all_exhausted'
    )

    # If we need only 30k rows 
    if len(interleaved_dataset) > 30000:
        interleaved_dataset = interleaved_dataset.select(range(30000))

    # Save each version to a different directory
    interleaved_dataset.save_to_disk(f"{base_save_path}variation_{i+1}")