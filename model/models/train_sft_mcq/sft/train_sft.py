import os
import sys
from enum import Enum

import torch
import time
from transformers import set_seed, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

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
push_to_hub = True #set True to push the model to the hub at the end of training
commit_message = "First model, trained on 40k examples for 3 epochs" #commit message
repo_id = "TeachersHateChatbots"
##################################################################
ANSWER_TEMPLATE = """
### EXPLANATION
<explanation>

### ANSWER
<correct_option>
"""


def get_mcq_options(samples):
    """
    Returns a dataset in the format:
    {
        "question": [str],
        "options": [list[str]],
        "correct_option": [str]
        "explanation": [str]
    }
    """
    raise NotImplementedError


def create_datasets(args, tokenizer):
    def apply_template(samples):
        system = {"role": "system", "content": "You are a helpful EPFL chatbot."}
        # texts = []
        messages = []

        for i, question in enumerate(samples["question"]):
            conversation = [system]
            conversation.append({"role": "user", "content": question})
            
            answer =  ANSWER_TEMPLATE.replace("<explanation>", samples["explanation"][i]).replace("<correct_option>", samples["correct_option"][i])
            
            conversation.append({"role": "assistant", "content": answer})

            # texts.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False))

            messages.append(conversation)

        # return {"text": texts}
        return {"messages": messages}

    dataset = load_dataset("json", data_files=args.dataset_name, split="train")


    if args.sample_quantity:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.sample_quantity))

    dataset = dataset.train_test_split(test_size=0.1)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.map(get_mcq_options, batched=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(get_mcq_options, batched = True, remove_columns=test_dataset.column_names)


    train_dataset = train_dataset.map(apply_template, batched=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(apply_template, batched=True, remove_columns=test_dataset.column_names)

    return train_dataset, test_dataset

def load_model(args):
    #TODO: decide how to load the model:
    # 1. Load the model with the finetuned adapter and firther finetune that adapter
    # 2. Load the model and adapter, merge them and unload adapter, and train new adapter on top of the already finetuned one
    # 3. Load the model and do the adapter thing they suggest for DPO (see DPOTRainer HuggingFace page, Option 3)

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
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.effective_batch_size // 2,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        # weight_decay=5e-5,
        # warmup_ratio=0.0,
        lr_scheduler_type="cosine",
        max_length=2048,
        num_train_epochs=3,
        logging_steps=5,
        log_level="info",
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        optim="paged_adamw_32bit",
        warmup_steps=5,
        max_grad_norm=1.0,
        resume_from_checkpoint=args.resume_from_checkpoint,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )

    print("\n\nTRAINING ARGS")
    print()
    print(training_args)
    print()
    print()
    print()

    if args.train_on_completions_only:
        #TODO: test this
        #TODO: I think we do not want to pass the `instruction_template` argument as altough our model is trained as conversational, we want QA, not to keep a dialog
        response_template = tokenizer.bos_token + "assistant\n"
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)
    else:
        collator=None


    #TODO: choose how to load the model
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        # dataset_text_field="text",
        peft_config=peft_config,
        data_collator=collator
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
    parser.add_argument("--dataset_name", type=str, help="Dataset for training")
    parser.add_argument("--sample_quantity", type=int, default=None, help="Number of entries to sample")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--model_name", type=str, default="rhysjones/phi-2-orange-v2", help="Base model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint to resume training from")
    parser.add_argument("--output_dir", type=str, help="Directory to output the model")
    parser.add_argument("--effective_batch_size", type=int, default=32, help="Effective batch size")
    parser.add_argument("--train_on_completions_only", type=bool, default=False, help="Whether to train on completions only")
    parser.add_argument("--packing", type=bool, default=False, help="Wether to use packing or not")


    args = parser.parse_args()

    if args.train_on_completions_only and args.packing:
        raise ValueError("Training on completions only cannot be used with packing")

    print(args.dataset_name)
    start = time.time()
    main(args)
    end = time.time()
    print(f"\033[92mTraining took {time.strftime('%H:%M:%S', time.gmtime(end - start))} (hours:minutes:seconds)\033[0m")