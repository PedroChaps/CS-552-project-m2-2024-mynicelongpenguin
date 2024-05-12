#! /bin/bash -l
module load gcc python

virtualenv --system-site-packages ~/venvs/test_env

echo "Created Virtual Environment"

source ~/venvs/test_env/bin/activate

echo "Installing Depenedencies..."

pip install --upgrade pip

pip3 install --no-cache-dir -r requirements.txt

echo "Dependencies installed"
echo "DONE"
deactivate
