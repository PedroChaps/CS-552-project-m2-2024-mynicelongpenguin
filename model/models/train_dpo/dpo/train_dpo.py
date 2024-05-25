import os
import sys
from enum import Enum

import torch
import time
from transformers import set_seed
from trl import DPOTrainer
from trl import DPOConfig

from datasets import DatasetDict, load_dataset, load_from_disk 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig
import argparse


##################################################################
# DEBUGGING
def dprint(*args, **kwargs):
    if True:
        print("[DEBUG train2.py]", *args, **kwargs)
##################################################################
# SOME PARAMETERS (may adjust in the code as well)
# username = "aloureir"

# model_name = "rhysjones/phi-2-orange-v2"

# dataset_name = f"/scratch/izar/{username}/project-m2-2024-mynicelongpenguin/data/combined_40k_train.jsonl"

# seed = 42
# resume_from_checkpoint = None #set to the path of the checkpoint to resume training from

# output_dir = "outputs/chosen_model"
push_to_hub = False #set True to push the model to the hub at the end of training
commit_message = None #commit message
repo_id = "TeachersHateChatbots"
##################################################################


def create_datasets(args, tokenizer):
    def apply_template(samples):
        system = {"role": "system", "content": "You are a helpful EPFL chatbot."}
        prompts = []
        chosens = []
        rejecteds = []

        for i, prompt in enumerate(samples["prompt"]):
            conversation = [system]
            conversation.append({"role": "user", "content": prompt})
            chosen = {"role": "assistant", "content": samples["chosen"][i]}
            rejected = {"role": "assistant", "content": samples["rejected"][i]}

            prompts.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False))
            chosens.append(tokenizer.apply_chat_template([chosen], tokenize=False, add_generation_prompt=False))
            rejecteds.append(tokenizer.apply_chat_template([rejected], tokenize=False, add_generation_prompt=False))

        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    dataset = load_dataset("json", data_files=args.dataset_name, split="train")


    if args.sample_quantity:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.sample_quantity))

    dataset = dataset.train_test_split(test_size=0.1)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.map(apply_template, batched=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(apply_template, batched=True, remove_columns=test_dataset.column_names)

    return train_dataset, test_dataset

def load_model(args):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            # quantization_config=bnb_config,
            trust_remote_code=True,
            # torch_dtype=torch.fp32,
            torch_dtype=torch.float32,
            device_map="auto",
        )


        # special_tokens = ChatmlSpecialTokens
        # chat_template = DEFAULT_CHATML_CHAT_TEMPLATE

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        dprint(tokenizer.pad_token)
        dprint(tokenizer.eos_token)
        dprint(tokenizer.bos_token)
        dprint(tokenizer.chat_template)


        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )

        return model, tokenizer, peft_config
    



def main(args):
    
    # Set seed for reproducibility
    set_seed(args.seed)
    # model
    model, tokenizer, peft_config = load_model(args)

    model.config.use_cache = False

    print(f"{model.device = }")

    train_dataset, test_dataset = create_datasets(args,tokenizer)

    print(f"{train_dataset[0] = }")
    training_args = DPOConfig(
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        loss_type="robust",
        label_pad_token_id=tokenizer.pad_token_id,
        padding_value=tokenizer.pad_token,
        max_length=2048,
        max_prompt_length=1024,
        max_target_length=1024,
        precompute_ref_log_probs=True,
        output_dir=args.output_dir,
        num_train_epochs=3,
        logging_steps=5,
        log_level="info",
        logging_strategy="steps",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        optim="paged_adamw_32bit",
        # optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        # weight_decay=5e-5,
        # warmup_ratio=0.0,
        warmup_steps=5,
        max_grad_norm=1.0,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_accumulation_steps=args.effective_batch_size // 2,
        gradient_checkpointing=True,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )

    print("\n\nTRAINING ARGS")
    print()
    print(training_args)
    print()
    print()
    print()

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()

    if push_to_hub:
        trainer.model.push_to_hub(commit_message=commit_message, repo_id=repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for Robust DPO")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta for DPO")
    parser.add_argument("--dataset_name", type=str, help="Dataset for training")
    parser.add_argument("--sample_quantity", type=int, default=None, help="Number of entries to sample")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--model_name", type=str, default="rhysjones/phi-2-orange-v2", help="Base model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint to resume training from")
    parser.add_argument("--output_dir", type=str, help="Directory to output the model")
    parser.add_argument("--effective_batch_size", type=int, default=32, help="Effective batch size")

    args = parser.parse_args()
    print(args.dataset_name)
    start = time.time()
    main(args)
    end = time.time()
    print(f"\033[92mTraining took {time.strftime('%H:%M:%S', time.gmtime(end - start))} (hours:minutes:seconds)\033[0m")