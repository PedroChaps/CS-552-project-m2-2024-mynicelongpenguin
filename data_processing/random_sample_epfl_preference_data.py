"""
This script processes a M1 preference pairs. Each entry in the dataset contains a prompt and associated preferred and rejected responses. 
The script loads data, organize it by prompt, perform a train-test split ensuring no prompt overlaps between sets, 
sample the data to limit the number of preference pairs per unique prompt in the training set, and finally, write the processed data back to JSONL.

Key Functions:
- load_data: Reads a JSONL file and loads it into a list of dictionaries.
- organize_data: Groups preference pairs by their prompts, filtering out entries with responses shorter than 10 characters. (some dude just wrote this: "chosen": "..." and "rejected": "...")
- train_test_split: Splits the organized data into training and testing sets based on a given proportion, ensuring all preference pairs for a prompt stay together.
- subsample_data: Reduces the number of preference pairs per prompt in the dataset to a specified maximum.
- write_output: Writes the processed data back to a JSONL file, maintaining the original format.
- main: Orchestrates the loading, organizing, splitting, subsampling, and output writing using specified file paths and parameters.

Usage:
Run the script with appropriate file paths and parameters to process the dataset into training and testing sets, with options to limit the data volume in the training set.
"""

import json
from collections import defaultdict
import random


def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def organize_data(data):
    organized_data = defaultdict(list)
    for entry in data:
        prompt = entry['prompt']
        chosen = entry['chosen']
        rejected = entry['rejected']

        if len(chosen) < 10 or len(rejected) < 10:
            continue

        pair = {'chosen': entry['chosen'], 'rejected': entry['rejected']}
        organized_data[prompt].append(pair)
    return organized_data


def train_test_split(organized_data, test_size=0.2):
    prompts = sorted(organized_data.keys())  # Sort prompts alphabetically
    total_prompts = len(prompts)
    split_index = int(total_prompts * (1 - test_size))  # Calculate initial index for split based on test size
    
    # Adjust split_index to ensure all entries of a unique prompt stay together
    if split_index < total_prompts:
        current_prompt = prompts[split_index]
        # Move split index forward if it's not at the start of a new unique prompt
        while split_index < total_prompts and prompts[split_index] == current_prompt:
            split_index += 1

    train_prompts = set(prompts[:split_index])  # Assign the first part to train set
    test_prompts = set(prompts[split_index:])  # Assign the remaining part to test set
    
    train_data = {prompt: pairs for prompt, pairs in organized_data.items() if prompt in train_prompts}
    test_data = {prompt: pairs for prompt, pairs in organized_data.items() if prompt in test_prompts}
    return train_data, test_data


def subsample_data(organized_data, max_pairs):
    subsampled_data = {}
    for prompt, pairs in organized_data.items():
        if len(pairs) > max_pairs:
            subsampled_data[prompt] = random.sample(pairs, max_pairs)
        else:
            subsampled_data[prompt] = pairs
    return subsampled_data


def write_output(data, output_file_path):
    with open(output_file_path, 'w') as file:
        for prompt, pairs in data.items():
            for pair in pairs:
                entry = {
                    'prompt': prompt,
                    'chosen': pair['chosen'],
                    'rejected': pair['rejected']
                }
                json.dump(entry, file)
                file.write('\n')


def main(input_file_path, train_output_path, test_output_path, max_pairs, test_size):
    data = load_data(input_file_path)
    organized_data = organize_data(data)
    
    # Train-test split
    train_data, test_data = train_test_split(organized_data, test_size)
    
    # Subsample train data
    train_data = subsample_data(train_data, max_pairs)
    # Optionally, subsample test data or keep it as is (avg. 17 pairs per unique prompt)
    # test_data = subsample_data(test_data, max_pairs)  # Uncomment this if you want to limit test data as well

    # Check for any overlapping prompts in train and test sets
    train_prompts = set(train_data.keys())
    test_prompts = set(test_data.keys())
    overlapping_prompts = train_prompts.intersection(test_prompts)
    if overlapping_prompts:
        print("Warning: There are overlapping prompts in train and test sets:", overlapping_prompts)
    else:
        print("No overlapping prompts found. Data split is clean.")

    # Write train and test data
    write_output(train_data, train_output_path)
    write_output(test_data, test_output_path)

if __name__ == '__main__':
    # Parameters
    input_file_path = '../data/dpo_formatted_epfl_preference_data.jsonl'
    train_output_path = '../data/M1_preference_data_train.jsonl'
    test_output_path = '../data/M1_preference_data_test.jsonl'
    max_pairs = 10  # limit to 10 pairs per prompt
    test_size = 0.2  # 20% of data for testing

    # Run the main function
    main(input_file_path, train_output_path, test_output_path, max_pairs, test_size)

