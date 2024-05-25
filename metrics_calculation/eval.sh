#!/bin/bash -l
#SBATCH --chdir /scratch/izar/aloureir/project-m2-2024-mynicelongpenguin/metrics_calculation 
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --time 10:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552


module load gcc python
module load cuda


# You only need to create this virtualenv once
# Feel free to replace the name "course_py-3.10" with your own environemnt name
# virtualenv --system-site-packages ~/venvs/mnlp2

# Activate the virtualenv everytime before you run the job
# Make sure the name matches the one you created
source ~/venvs/mnlp2/bin/activate


# upgrade pip the first time you load the environment
# pip install --upgrade pip


# Only when you need to update any packages you already installed
# pip3 install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# pip3 install -r ./train_dpo/requirements.txt
# pip3 install -U git+https://github.com/huggingface/trl


echo "Install Dependency completed"

echo "Going to run the program..."

python3 eval_loss.py \
    --model_path_or_name "/scratch/izar/aloureir/project-m2-2024-mynicelongpenguin/model/models/outputs/combined_40k_model" \
    --dataset_path "/scratch/izar/aloureir/project-m2-2024-mynicelongpenguin/data/combined_test.jsonl" \
    --base_model_name "rhysjones/phi-2-orange-v2" \
    --sample_quantity 5000
    

echo "Test complete on $(hostname)"
sleep 2

deactivate
