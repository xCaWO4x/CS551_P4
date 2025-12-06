#!/usr/bin/env bash
# Simple test job to verify the setup works
#SBATCH --time=0-1:0:0
#SBATCH --partition=academic
#SBATCH --job-name=test_cs551
#SBATCH --output=test_slurm-%j.out
#SBATCH --error=test_slurm-%j.err
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
# Initialize conda
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate myenv
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
echo Testing CS551_P4 setup

# Run a quick test with minimal parameters
python main.py -a PPO --ppo_lr 0.0001 --max_steps 128 --n_updates 3 --batch_size 32 --gamma 0.99 --clip 0.01 --ent_coeff 0.0 --n_trials 1 --epoch 10

echo Test job completed

