import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PhiForCausalLM,
    set_seed
)
from peft import PeftModel, PeftConfig
from tqdm import tqdm
seed = 42
def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    if isinstance(model, PhiForCausalLM):
        print("MODEL IS OF PHIFORCAUSALLM")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    # peft_config = PeftConfig.from_pretrained(args.model_path_or_name)

    model = PeftModel.from_pretrained(model, args.model_path_or_name)

    model = model.merge_and_unload()

    return model, tokenizer


def load_datasets(args, tokenizer):
    system = {"role": "system", "content": "You are a helpful EPFL chatbot."}
    def choose_chosen(samples):

        # prompts = []
        prompt_and_answers = []

        for i, prompt in enumerate(samples["prompt"]):
            # prompts.append(tokenizer.apply_chat_template([system, {"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))
            prompt_and_answers.append(tokenizer.apply_chat_template([system, {"role": "user", "content": prompt}, {"role": "assistant", "content": samples["chosen"][i]}], tokenize=False, add_generation_prompt=False))

        # return {"eval_text": prompts, "ground_truth": prompt_and_answers}
        return {"ground_truth": prompt_and_answers}


    dataset = load_dataset('json', data_files=args.dataset_path, split="train")

    if args.sample_quantity:
        dataset = dataset.shuffle(seed=seed).select(range(args.sample_quantity))

    print("Length of vanilla dataset:", len(dataset))
    dataset = dataset.map(
        choose_chosen,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=8
    )
    print("Length of processed dataset:", len(dataset))

    return dataset


def calculate_losses(model, tokenizer, dataset, args):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(total=len(dataloader), desc="Eval")
    for i, batch in enumerate(dataloader):
        tokenized_batch = tokenizer(batch['ground_truth'], padding="max_length", truncation=True, return_tensors='pt', max_length=2048)

        inputs = tokenized_batch.input_ids.to(model.device)
        attention_mask = tokenized_batch.attention_mask.to(model.device)

        #set pad tokens in labels to -100 so they are masked and do not count towards loss
        labels = torch.where(inputs == tokenizer.pad_token_id, -100, inputs)

        with torch.no_grad():
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        total_loss += loss.item()
        num_batches += 1

        if i % 10 == 0:
            progress_bar.update(10)
            print(f"Batch {i} loss: {loss.item()}")
        
    
    return total_loss / num_batches

    

def main():
    set_seed(seed)
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path_or_name', type=str, help='Model path if saved or model name if from HF')
    parser.add_argument('--dataset_path', type=str, help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size to be used for inference')
    parser.add_argument('--base_model_name', type=str, help='Base model name to load PEFT config')
    parser.add_argument('--sample_quantity', type=int, default=None, help="Number of samples to use for evaluation, None for all dataset")
    args = parser.parse_args()

    print("ARGS")
    print(args)
    print()
    print()

    model, tokenizer = load_model(args)

    dataset = load_datasets(args, tokenizer)


    avg_loss = calculate_losses(model, tokenizer, dataset, args)

    print("Average Loss:", avg_loss)

    with open(f"eval_loss_{args.model_path_or_name.split('/')[-1]}.txt", "w") as f:
        f.write(str(avg_loss))



if __name__ == "__main__":
    main()

    


