#! /bin/bash -l
module load gcc python cuda

echo "Loading virtualenv..."

source ~/venvs/venv_evaluator/bin/activate

echo "Going to run..."

python evaluator.py

echo "Evaluatior run"

deactivate
