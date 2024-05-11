#!/bin/bash -l
#SBATCH --chdir /scratch/izar/chaparro
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --time 15:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552

module load gcc python
# You only need to create this virtualenv once
# Feel free to replace the name "course_py-3.10" with your own environemnt name
virtualenv --system-site-packages ~/venvs/mnlp

# Activate the virtualenv everytime before you run the job
# Make sure the name matches the one you created
source ~/venvs/mnlp/bin/activate

# upgrade pip the first time you load the environment
pip install --upgrade pip

# For the first time, you need to install the dependencies
# You don't have to do the installation for every job.
# Only when you need to update any packages you already installed
pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip3 install -r ./train_sft/requirements.txt

echo "Install Dependency completes"

echo "Testing Peft Example ..."


# First argument ... --output_dir
# I think we're changing those everytime 

# --num_train_epochs ... --dataset_text_field "content"
# Can be changed, left this values for now as they are the "default ones"
# --splits "train" can be "train,test" as well

# --seed 100 ... --chat_template_format "none"
# Others that can be changed but prob. not often

python3 ./train_sft/sft/train.py \
    --model_name_or_path "microsoft/phi-2" \
    --dataset_name "train_sft/data/MMLU-STEM_final" \
    --output_dir "outputs/THC-conf1" \
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
        --max_seq_len 64 \
        --splits "train" \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --dataset_text_field "content" \
    --seed 100 \
    --add_special_tokens False \
    --append_concat_token False \
    --use_reentrant True \
    --chat_template_format "none" \
    
    # For LoRA
    # --use_peft_lora True \
    # --lora_r 8 \
    # --lora_alpha 16 \
    # --lora_dropout 0.1 \
    #--lora_target_modules "all-linear" \
    

echo "Test complete on $(hostname)"
sleep 2
