#!/usr/bin/env bash
#SBATCH --time=1-23:0:0
#SBATCH --partition=academic
#SBATCH --job-name=cs551_p4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
# Initialize conda
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate myenv
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
echo Command: $@

exec $@

