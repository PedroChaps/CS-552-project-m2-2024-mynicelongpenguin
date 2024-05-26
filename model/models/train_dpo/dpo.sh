#!/bin/bash -l
#SBATCH --chdir /scratch/izar/aloureir/project-m2-2024-mynicelongpenguin/model/models
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --time 71:59:59
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552


module load gcc python
module load cuda


SAMPLE=3000


# You only need to create this virtualenv once
# Feel free to replace the name "course_py-3.10" with your own environemnt name
# virtualenv --system-site-packages ~/venvs/mnlp2

# Activate the virtualenv everytime before you run the job
# Make sure the name matches the one you created
source ~/venvs/mnlp2/bin/activate

# huggingface-cli login --token $HF_TOKEN --add-to-git-credential
huggingface-cli login --token $HF_TOKEN 

# upgrade pip the first time you load the environment
# pip install --upgrade pip


# Only when you need to update any packages you already installed
# pip3 install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# pip3 install -r ./train_dpo/requirements.txt
# pip3 install -U git+https://github.com/huggingface/trl


echo "Install Dependency completed"

echo "Going to run the program..."

python3 ./train_dpo/dpo/train_dpo.py --lr 1e-5 \
    --beta 0.3 \
    --label_smoothing 0.1 \
    --dataset_name "/scratch/izar/aloureir/project-m2-2024-mynicelongpenguin/data/combined_40k_train.jsonl" \
    --output_dir "/scratch/izar/aloureir/project-m2-2024-mynicelongpenguin/model/models/outputs/final_model_40k" \
    --effective_batch_size 32


echo "Test complete on $(hostname)"
sleep 2

deactivate
