#! /bin/bash -l
module load gcc python

echo "Loading virtualenv..."

source ~/venvs/test_env/bin/activate

echo "Going to run..."

python evaluator.py

echo "Evaluatior run"
echo "DONE"

deactivate
