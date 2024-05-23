#!/bin/bash -l
#SBATCH --chdir /scratch/izar/aloureir/project-m2-2024-mynicelongpenguin/model/models
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552

module load gcc python
module load nvhpc
module load cuda
# You only need to create this virtualenv once
# Feel free to replace the name "course_py-3.10" with your own environemnt name
# virtualenv --system-site-packages ~/venvs/mnlp

# Activate the virtualenv everytime before you run the job
# Make sure the name matches the one you created
source ~/venvs/mnlp/bin/activate

# upgrade pip the first time you load the environment
# pip install --upgrade pip

# For the first time, you need to install the dependencies
# You don't have to do the installation for every job.
# Only when you need to update any packages you already installed
# pip3 install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# pip3 install -r ./train_sft/requirements.txt

echo "Installing flash attention..."
# pip3 install packaging
# pip3 install ninja
# pip3 install flash-attn --no-build-isolation

echo "Install Dependency completes"

echo "Testing Peft Example ..."


# First argument ... --output_dir
# I think we're changing those everytime 

# --num_train_epochs ... --dataset_text_field "content"
# Can be changed, left this values for now as they are the "default ones"
# --splits "train" can be "train,test" as well

# --seed 100 ... --chat_template_format "none"
# Others that can be changed but prob. not often

    # --model_name_or_path "unsloth/Phi-3-mini-4k-instruct" \
python3 ./train_dpo/dpo/train.py \
    --model_name_or_path "rhysjones/phi-2-orange-v2" \
    --dataset_name "../datasets/dpo_preference_example" \
    --dataset_is_json True \
    --output_dir "outputs/THC-conf2-AndreIsJustTesting" \
        --num_train_epochs 3 \
        --logging_steps 5 \
        --log_level "info" \
        --logging_strategy "steps" \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --learning_rate 1e-4 \
        --lr_scheduler_type "cosine" \
        --weight_decay 1e-4 \
        --warmup_ratio 0.0 \
        --max_grad_norm 1.0 \
        --max_seq_len 2048 \
        --splits "train,test" \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 32 \
        --dataset_text_field "text" \
        --use_flash_attn False \
        --beta 0.1 \
    --seed 100 \
    --add_special_tokens False \
    --append_concat_token False \
    --use_reentrant True \
    --use_unsloth False \
    --chat_template_format "chatml" \
        --use_peft_lora True \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --lora_target_modules "all-linear" \
    --use_4bit_quantization True \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_type "nf4"
    # For LoRA
    

echo "Test complete on $(hostname)"
sleep 2

deactivate
