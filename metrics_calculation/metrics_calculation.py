from datasets import load_dataset, load_from_disk, concatenate_datasets
import gpt_wrapper
from gpt_wrapper.chat import Chat
import tqdm
import random
import os
import evaluate


from bert_score import BERTScorer

import argparse


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def calculate_COMMET_metric(chosen_dataset, model_outputs):
    ...

def calculate_BERTScore(dataset):
    
    def calculate_BERTScore_for_row(row):
        
        answer1 = row["answer1"]
        answer2 = row["answer2"]
        
        scorer = BERTScorer(model_type="bert-base-uncased")
        P, R, F1 = scorer.score([answer1], [answer2])
        
        row["BERTScore_P"] = P[0]
        row["BERTScore_R"] = R[0]
        row["BERTScore_F1"] = F1[0]
        
        return row
    
    
    dataset = dataset.map(lambda x: calculate_BERTScore_for_row(x), num_proc=32)
    
    # Does statistics on the dataset i.e. prints the average values
    total_P, total_R, total_F1, total_entries = 0, 0, 0, 0
    for entry in dataset:
        total_P += entry["BERTScore_P"]
        total_R += entry["BERTScore_R"]
        total_F1 += entry["BERTScore_F1"]
        total_entries += 1
    
    print(f"{color.BOLD}### BERTScore ###{color.END}")
    print(f"{color.BOLD}Total:{color.END}{color.GREEN} {total_entries} entries {color.END}")
    print(f"{color.BOLD}Average BERTScore P:{color.END}{color.GREEN} {total_P / total_entries}{color.END}")
    print(f"{color.BOLD}Average BERTScore R:{color.END}{color.GREEN} {total_R / total_entries}{color.END}")
    print(f"{color.BOLD}Average BERTScore F1:{color.END}{color.GREEN} {total_F1 / total_entries}{color.END}")
    print()
    
    return dataset
    

def calculate_BLEU(dataset):

    bleu = evaluate.load('bleu')

    def calculate_BLEU_for_row(row):
        
        answer1 = row["answer1"]
        answer2 = row["answer2"]
        
        bleu_val = bleu.compute(predictions=[answer1], references=[answer2])["bleu"]
        
        row["BLEU"] = bleu_val
    
        return row
    
    
    dataset = dataset.map(lambda x: calculate_BLEU_for_row(x), num_proc=32)
    
    # Does statistics on the dataset i.e. prints the average BLEU value
    total_BLEU, total_entries = 0, 0
    for entry in dataset:
        total_BLEU += entry["BLEU"]
        total_entries += 1
    
    print(f"{color.BOLD}### BLEU ###{color.END}")
    print(f"{color.BOLD}Total:{color.END}{color.GREEN} {total_entries} entries {color.END}")
    print(f"{color.BOLD}Average BLEU:{color.END}{color.GREEN} {total_BLEU / total_entries}{color.END}")
    print()
    
    return dataset



def calculate_ROUGE(dataset):

    rouge = evaluate.load('rouge')

    def calculate_ROUGUE_for_row(row):
        
        answer1 = row["answer1"]
        answer2 = row["answer2"]
        
        rouge_vals = rouge.compute(predictions=[answer1], references=[answer2])

        row["ROUGE-1"] = rouge_vals["rouge1"]
        row["ROUGE-2"] = rouge_vals["rouge2"]
        row["ROUGE-L"] = rouge_vals["rougeL"]
    
        return row
    
    
    dataset = dataset.map(lambda x: calculate_ROUGUE_for_row(x), num_proc=32)
    
    # Does statistics on the dataset i.e. prints the average ROUGE values
    total_ROUGE_1, total_ROUGE_2, total_ROUGE_L, total_entries = 0, 0, 0, 0
    for entry in dataset:
        total_ROUGE_1 += entry["ROUGE-1"]
        total_ROUGE_2 += entry["ROUGE-2"]
        total_ROUGE_L += entry["ROUGE-L"]
        total_entries += 1
    
    print(f"{color.BOLD}### ROUGE ###{color.END}")
    print(f"{color.BOLD}Total:{color.END}{color.GREEN} {total_entries} entries {color.END}")
    print(f"{color.BOLD}Average ROUGE-1:{color.END}{color.GREEN} {total_ROUGE_1 / total_entries}{color.END}")
    print(f"{color.BOLD}Average ROUGE-2:{color.END}{color.GREEN} {total_ROUGE_2 / total_entries}{color.END}")
    print(f"{color.BOLD}Average ROUGE-L:{color.END}{color.GREEN} {total_ROUGE_L / total_entries}{color.END}")
    print()
    
    return dataset




def get_merged_dataset(args):
    
    assert os.path.exists(args.file1) and os.path.exists(args.file2)
    assert args.file1.endswith(".jsonl") and args.file2.endswith(".jsonl")
    
    file1 = load_dataset("json", data_files=args.file1, split="train")
    file2 = load_dataset("json", data_files=args.file2, split="train")

    assert len(file1) == len(file2)

    for i in range(len(file2)):        
        assert "question" in file1[i] and "question" in file2[i]
        assert "answer" in file1[i] and "answer" in file2[i]
        assert file1[i]["question"] == file2[i]["question"]

    # Creates a dataset that is a merge of both so the .map() method can be applied to the dataset
    file1 = file1.map(lambda x: {"answer1": x["answer"]}).remove_columns("answer")
    file2 = file2.map(lambda x: {"answer2": x["answer"]}).remove_columns(["answer", "question"])
    
    dataset = concatenate_datasets([file1, file2], axis=1)
    return dataset
    
    

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--file1", type=str, required=True, help="One of the files with inferences")
    parser.add_argument("--file2", type=str, required=True, help="The other file with inferences")
    parser.add_argument("--output", type=str, required=True, help="The file to save the output to")

    args = parser.parse_args()
    
    dataset = get_merged_dataset(args)
    
    dataset = calculate_ROUGE(dataset)
    dataset = calculate_BLEU(dataset)
    dataset = calculate_BERTScore(dataset)
    # Call other metrics here
    
    # saves the dataset to disk
    dataset.to_json(args.output)



if __name__ == "__main__":
    main()