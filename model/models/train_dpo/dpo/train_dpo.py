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


##################################################################
# DEBUGGING
def dprint(*args, **kwargs):
    if True:
        print("[DEBUG train2.py]", *args, **kwargs)
##################################################################
# SOME PARAMETERS (may adjust in the code as well)
username = "aloureir"

model_name = "rhysjones/phi-2-orange-v2"
new_model_name = "MNLP_try"

dataset_name = f"/scratch/izar/{username}/project-m2-2024-mynicelongpenguin/model/datasets/dpo_preference_example.jsonl"

seed = 42
resume_from_checkpoint = None #set to the path of the checkpoint to resume training from
##################################################################


def create_datasets(tokenizer):
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

            prompts.append(tokenizer.apply_chat_template(conversation, tokenize=False))
            chosens.append(tokenizer.apply_chat_template([chosen], tokenize=False))
            rejecteds.append(tokenizer.apply_chat_template([rejected], tokenize=False))
            
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    dataset = load_dataset("json", data_files=dataset_name, split="train")
    dataset = dataset.train_test_split(test_size=0.1)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_dataset = train_dataset.map(apply_template, batched=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(apply_template, batched=True, remove_columns=test_dataset.column_names)

    return train_dataset, test_dataset

def load_model():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # quantization_config=bnb_config,
            trust_remote_code=True,
            # torch_dtype=torch.fp32,
            torch_dtype=torch.float32,
            device_map="auto",
        )


        # special_tokens = ChatmlSpecialTokens
        # chat_template = DEFAULT_CHATML_CHAT_TEMPLATE

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        dprint(tokenizer.pad_token)
        dprint(tokenizer.eos_token)
        dprint(tokenizer.bos_token)
        dprint(tokenizer.chat_template)


        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )

        return model, tokenizer, peft_config
    



def main():
    
    # Set seed for reproducibility
    set_seed(42)

    # model
    model, tokenizer, peft_config = load_model()

    model.config.use_cache = False

    print(f"{model.device = }")

    train_dataset, test_dataset = create_datasets(tokenizer)

    print(f"{train_dataset[0] = }")

    training_args = DPOConfig(
        beta=0.1,
        label_pad_token_id=tokenizer.pad_token_id,
        padding_value=tokenizer.pad_token,
        max_length=2048,
        max_prompt_length=1024,
        max_target_length=1024,
        precompute_ref_log_probs=True,
        output_dir="outputs/try2",
        num_train_epochs=3,
        logging_steps=5,
        log_level="info",
        logging_strategy="steps",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        # optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
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

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\033[92mTraining took {time.strftime('%H:%M:%S', time.gmtime(end - start))} (hours:minutes:seconds)\033[0m")