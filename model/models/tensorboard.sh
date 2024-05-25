#! /bin/bash

module load gcc python openmpi py-tensorflow
#module load cuda

ipnport=$(shuf -i8000-9999 -n1)

#TENSOR_DIR=$1
TENSOR_DIR=/scratch/izar/aloureir/project-m2-2024-mynicelongpenguin/model/models/outputs/combined_40k_model/runs/May24_22-49-14_i28
tensorboard --logdir ${TENSOR_DIR}  --port=${ipnport} --bind_all
