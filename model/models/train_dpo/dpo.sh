#!/bin/bash -l
#SBATCH --chdir /scratch/izar/aloureir/project-m2-2024-mynicelongpenguin/model/models
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --time 2:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552

module load gcc python
module load cuda

# You only need to create this virtualenv once
# Feel free to replace the name "course_py-3.10" with your own environemnt name
# virtualenv --system-site-packages ~/venvs/mnlp

# Activate the virtualenv everytime before you run the job
# Make sure the name matches the one you created
source ~/venvs/mnlp/bin/activate

# upgrade pip the first time you load the environment
# pip install --upgrade pip


# Only when you need to update any packages you already installed
# pip3 install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# pip3 install -r ./train_sft/requirements.txt
# pip install -U git+https://github.com/huggingface/trl

echo "Install Dependency completed"

echo "Going to run the program..."

python3 ./train_dpo/dpo/train_dpo.py
    

echo "Test complete on $(hostname)"
sleep 2

deactivate
