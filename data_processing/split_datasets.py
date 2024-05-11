from datasets import load_dataset, load_from_disk
from tqdm import tqdm

def main():
    for dataset_name in tqdm(["epfl_all_formatted", "MMLU-STEM_processed", "open_platypus_justified_2", "theoremqa_justified_formatted_2", "arxiv-physics-instruct-tune-30k-formatted"]):
        
        path = "data/" + dataset_name
        
        dataset = load_from_disk(path)
        #print(dataset)
        
        # Splits the dataset into two datasets, a test set and a training set
        if dataset_name == "arxiv-physics-instruct-tune-30k-formatted":
            splitted_dataset = dataset["train"].train_test_split(test_size=0.2)
        else:
            splitted_dataset = dataset.train_test_split(test_size=0.2)
        
        # Save the datasets to disk
        splitted_dataset.save_to_disk(path + "_split")


if __name__ == "__main__":
    main()