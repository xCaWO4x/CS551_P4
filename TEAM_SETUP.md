# Quick Setup Guide for CS551_P4

## Environment Setup

### 1. Create/Activate Conda Environment

```bash
# Check if myenv exists
conda env list

# If it doesn't exist, create it:
conda create -n myenv python=3.11 -y
conda activate myenv

# Install dependencies
pip install torch numpy matplotlib gymnasium
pip install gymnasium[box2d]  # Required for BipedalWalker
```

### 2. Verify Installation

```bash
conda activate myenv
python -c "import torch; import gymnasium; env = gymnasium.make('BipedalWalker-v3'); print('Setup complete!')"
```

## Running Jobs

### Single Job

```bash
sbatch runner.sh python main.py -a PPO --ppo_lr 0.0001 --max_steps 256 --n_updates 5 --batch_size 64 --n_trials 3
```

### Hyperparameter Search

```bash
bash search.sh  # Submits 810 jobs - use with caution!
```

### Other Algorithms

```bash
# ESPPO
sbatch runner.sh python main.py -a ESPPO --population_size 10 --sigma 0.1 --n_seq 1 --ppo_lr 0.0001 --n_trials 3

# MAXPPO
sbatch runner.sh python main.py -a MAXPPO --population_size 10 --sigma 0.1 --ppo_lr 0.0001 --n_trials 3

# ALTPPO
sbatch runner.sh python main.py -a ALTPPO --population_size 10 --n_alt 5 --ppo_lr 0.0001 --n_trials 3
```

## Script Configuration

- **Partition**: `academic` (edit `runner.sh` if needed)
- **Environment**: `myenv` (must exist in your conda)
- **Output**: `slurm-<jobid>.out` and `slurm-<jobid>.err`

## Common Issues

**"Could not find conda environment: pytorch"**
- Scripts use `myenv`, not `pytorch`. Create it (see Step 1).

**"ModuleNotFoundError: No module named 'Box2D'"**
- Run: `pip install gymnasium[box2d]` in your `myenv` environment.

**"Invalid account or partition"**
- Check available partitions: `sinfo`
- Edit `#SBATCH --partition=academic` in `runner.sh` if needed.

## Quick Test

Before running large jobs, test with:

```bash
sbatch runner.sh python main.py -a PPO --ppo_lr 0.0001 --max_steps 128 --n_updates 3 --batch_size 32 --n_trials 1 --epoch 5
```

Check output: `cat slurm-<jobid>.out`

