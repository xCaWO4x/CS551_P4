# Running Experiments Guide

This guide covers how to set up and run experiments on the WPI Turing cluster using SLURM, including baseline configurations and hyperparameter search.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [SLURM Scripts Overview](#slurm-scripts-overview)
3. [Running Single Experiments](#running-single-experiments)
4. [Running Baseline Experiments](#running-baseline-experiments)
5. [Hyperparameter Search](#hyperparameter-search)
6. [Monitoring Jobs](#monitoring-jobs)
7. [Common Issues and Solutions](#common-issues-and-solutions)

## Environment Setup

### 1. Check Your Conda Installation

First, verify that conda is available:

```bash
which conda
# or
ls ~/miniconda3
```

If conda is not installed, you'll need to install it first.

### 2. Create/Activate the Conda Environment

The scripts expect a conda environment named `myenv`. Check if it exists:

```bash
conda env list
```

**If `myenv` doesn't exist**, create it and install dependencies:

```bash
# Create the environment (Python 3.11 recommended)
conda create -n myenv python=3.11 -y

# Activate the environment
conda activate myenv

# Install required packages
pip install torch numpy matplotlib gymnasium

# Install Box2D (required for BipedalWalker environment)
pip install gymnasium[box2d]

# Install tqdm (optional, for progress bars)
pip install tqdm
```

**If `myenv` already exists**, just activate it and verify packages:

```bash
conda activate myenv
python -c "import torch; import matplotlib; import gymnasium; print('All packages available!')"
```

If the Box2D import fails, install it:

```bash
pip install gymnasium[box2d]
```

### 3. Verify Installation

Test that everything works:

```bash
conda activate myenv
python -c "import torch; import numpy; import matplotlib; import gymnasium; env = gymnasium.make('BipedalWalker-v3'); print('Setup complete!')"
```

## SLURM Scripts Overview

The project includes several SLURM scripts:

- **`runner.sh`**: Template script for submitting individual jobs
- **`search.sh`**: Script that submits multiple jobs for hyperparameter search
- **`run_baseline.sh`**: Script to run baseline experiments with paper configurations

### Script Configuration

The scripts are configured with:
- **Partition**: `academic` (WPI Turing cluster partition)
- **Time limit**: 1 day 23 hours for `runner.sh`
- **Conda environment**: `myenv`
- **Output files**: `slurm-<jobid>.out` and `slurm-<jobid>.err`

**Note**: If you need to change the partition (e.g., to `short` or `long`), edit the `#SBATCH --partition=academic` line in `runner.sh`.

## Running Single Experiments

### Basic Command

To submit a single experiment:

```bash
sbatch runner.sh python main.py -a PPO --ppo_lr 0.0001 --max_steps 256 --n_updates 5 --batch_size 64 --gamma 0.99 --clip 0.01 --ent_coeff 0.0 --n_trials 3 --epoch 100
```

### Examples for Different Algorithms

```bash
# PPO
sbatch runner.sh python main.py -a PPO --ppo_lr 0.0001 --n_updates 5 --batch_size 64 --n_trials 3

# ESPPO
sbatch runner.sh python main.py -a ESPPO --population_size 10 --sigma 0.1 --n_seq 1 --ppo_lr 0.0001 --n_trials 3

# CMA-PPO
sbatch runner.sh python main.py -a CMA_PPO \
    --n_updates 5 \
    --batch_size 2048 \
    --max_steps 16000 \
    --gamma 0.99 \
    --lam 0.95 \
    --cma_lr_mean 3e-4 \
    --cma_lr_var 3e-4 \
    --cma_lr_value 1e-3 \
    --history_size 5 \
    --kernel_std 0.1 \
    --n_trials 5 \
    --epoch 10000

# MAXPPO
sbatch runner.sh python main.py -a MAXPPO --population_size 10 --sigma 0.1 --ppo_lr 0.0001 --n_trials 3

# ALTPPO
sbatch runner.sh python main.py -a ALTPPO --population_size 10 --n_alt 5 --ppo_lr 0.0001 --n_trials 3
```

### Quick Test

Before running large jobs, always test with a small job first:

```bash
# Quick test (should complete in < 1 minute)
sbatch runner.sh python main.py -a PPO --ppo_lr 0.0001 --max_steps 128 --n_updates 3 --batch_size 32 --n_trials 1 --epoch 5
```

Check the output to ensure everything works before submitting large jobs.

## Running Baseline Experiments

### Baseline Configuration

We established baseline experiments for **PPO** and **ES-PPO** algorithms using the exact hyperparameters from the original paper. These baselines serve as the reference point for comparing future improvements.

### PPO Baseline (Section 4.1.1 from Paper)

The PPO baseline uses the following hyperparameters:

- **Learning rate**: 0.0001 (chosen among {0.0001, 0.00025, 0.001})
- **Max steps (H)**: 256
- **Batch size**: 32
- **n_updates (l)**: 3 (iterations over the entire 256 samples)
- **Clip (ε)**: 0.2 (chosen among {0.01, 0.02, 0.2})
- **Entropy coefficient**: 0.0 (chosen among {0.0, 0.0001})
- **Gamma (γ)**: 0.99 (fixed)
- **Trials**: 5
- **Max iterations**: 10,000

**Command:**
```bash
python main.py -a PPO \
    --ppo_lr 0.0001 \
    --max_steps 256 \
    --batch_size 32 \
    --n_updates 3 \
    --gamma 0.99 \
    --clip 0.2 \
    --ent_coeff 0.0 \
    --n_trials 5 \
    --epoch 10000
```

### ES-PPO Baseline (Section 4.1.3 from Paper)

The ES-PPO baseline uses the same hyperparameters as PPO and ES, with one key difference:

- **Population size (k)**: 5 (chosen among {5, 10, 20})
- **Sigma (σ²)**: 0.1 (chosen among {0.1, 0.001, 0.0001})
- **ES learning rate**: 0.001 (chosen among {0.0001, 0.0025, 0.001})
- **PPO learning rate**: 0.0001
- **Max steps**: 256
- **Batch size**: 32
- **n_updates (l)**: 1 (different from standalone PPO - helps decrease variance)
- **Clip**: 0.2
- **Entropy coefficient**: 0.0
- **Gamma**: 0.99
- **n_seq**: 1
- **Trials**: 5
- **Max iterations**: 10,000

**Key difference from standalone PPO**: `n_updates=1` instead of `3`. The paper notes this helps decrease variance in updates despite worse sample efficiency.

**Command:**
```bash
python main.py -a ESPPO \
    --population_size 5 \
    --sigma 0.1 \
    --es_lr 0.001 \
    --ppo_lr 0.0001 \
    --max_steps 256 \
    --batch_size 32 \
    --n_updates 1 \
    --gamma 0.99 \
    --clip 0.2 \
    --ent_coeff 0.0 \
    --n_seq 1 \
    --n_trials 5 \
    --epoch 10000
```

### Running Baselines with Script

Use the provided script to run both baselines:

```bash
bash run_baseline.sh
```

This will submit two SLURM jobs:
- Job 1: PPO baseline
- Job 2: ES-PPO baseline

### Manual Submission

You can also submit jobs manually using `sbatch`:

```bash
# PPO baseline
sbatch runner.sh python main.py -a PPO --ppo_lr 0.0001 --max_steps 256 --batch_size 32 --n_updates 3 --gamma 0.99 --clip 0.2 --ent_coeff 0.0 --n_trials 5 --epoch 10000

# ES-PPO baseline
sbatch runner.sh python main.py -a ESPPO --population_size 5 --sigma 0.1 --es_lr 0.001 --ppo_lr 0.0001 --max_steps 256 --batch_size 32 --n_updates 1 --gamma 0.99 --clip 0.2 --ent_coeff 0.0 --n_seq 1 --n_trials 5 --epoch 10000
```

### Expected Runtime

Based on test runs:

- **PPO**: ~2-4 hours (5 trials × 10,000 iterations)
- **ES-PPO**: ~5-10 hours (5 trials × 10,000 iterations × population_size=5)

Note: Times may vary based on:
- Early stopping if reward goal (300) is reached
- Cluster load and GPU availability
- Specific hyperparameter configuration

### Results Location

Results are saved in `./checkpoints/` with algorithm-specific directory names:

- PPO: `./checkpoints/PPO_3_32_256_0.99_0.2_0.0_0.0001/`
- ES-PPO: `./checkpoints/ESPPO_5_0.1_1_32_256_0.99_0.2_0.0_0.001_0.0001_1/`

Each directory contains:
- `results.txt`: Summary with average final reward and time
- `rewards.txt`: Reward history
- `rewards_plot.png`: Training curve plot
- `weights.pkl`: Final policy weights
- `metrics.pkl`: Detailed metrics (KL divergence, clip fraction, etc.)

## Hyperparameter Search

### Running Hyperparameter Search

To run the full hyperparameter search grid (submits many jobs):

```bash
bash search.sh
```

**Warning**: This will submit a large number of jobs. Make sure you have sufficient cluster resources and quota.

The `search.sh` script searches over:
- **PPO**: Learning rates, max steps, n_updates, batch sizes, clip values, entropy coefficients
- **ESPPO**: All PPO parameters plus ES-specific parameters (population size, sigma, ES learning rate)

### Customizing Search

To customize the search, edit `search.sh` and modify the parameter ranges.

## Monitoring Jobs

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View specific job
squeue -j <jobid>

# View job details
scontrol show job <jobid>
```

### Viewing Output

Job output is saved to:
- **Standard output**: `slurm-<jobid>.out`
- **Standard error**: `slurm-<jobid>.err`

View output in real-time (if job is running):

```bash
tail -f slurm-<jobid>.out
```

View completed job output:

```bash
cat slurm-<jobid>.out
cat slurm-<jobid>.err
```

### Canceling Jobs

Cancel a specific job:

```bash
scancel <jobid>
```

Cancel all your jobs:

```bash
scancel -u $USER
```

## Common Issues and Solutions

### Issue 1: "EnvironmentNameNotFound: Could not find conda environment: pytorch"

**Solution**: The scripts use `myenv`, not `pytorch`. Make sure you've created the `myenv` environment (see Environment Setup).

### Issue 2: "ModuleNotFoundError: No module named 'Box2D'"

**Solution**: Install Box2D in your conda environment:

```bash
conda activate myenv
pip install gymnasium[box2d]
```

### Issue 3: "Invalid account or account/partition combination specified"

**Solution**: Check available partitions:

```bash
sinfo
```

Then edit `runner.sh` to use an available partition (e.g., change `academic` to `short` or `long`).

### Issue 4: Jobs stuck in queue

**Solution**: 
- Check partition availability: `sinfo`
- Try a different partition with more available nodes
- Reduce time limit if using `short` partition
- Check your account limits: `sacctmgr show assoc user=$USER`

### Issue 5: "activate: No such file or directory"

**Solution**: The scripts use modern conda activation. Make sure you have miniconda3 installed at `~/miniconda3`. If it's in a different location, update the path in `runner.sh`:

```bash
eval "$(~/path/to/conda/bin/conda shell.bash hook)"
```

### Issue 6: No output visible during long-running jobs

**Solution**: The code now includes real-time output flushing. Make sure you have:
- `tqdm` installed for progress bars: `pip install tqdm`
- Output files are being written: `tail -f slurm-<jobid>.out`

## Best Practices

### Resource Management

- **Start small**: Test with `--n_trials 1 --epoch 10` before running full experiments
- **Monitor resources**: Use `squeue` and `sinfo` to check cluster status
- **Clean up**: Remove old output files periodically to save disk space

### Experiment Organization

- **Use descriptive job names**: Edit `--job-name` in `runner.sh` for different experiments
- **Organize outputs**: Results are saved to `./checkpoints/` by default (configurable with `--directory`)
- **Track job IDs**: Keep a log of job IDs and their purposes

### Testing Before Large Runs

Always test your setup with a small job first:

```bash
sbatch runner.sh python main.py -a PPO --ppo_lr 0.0001 --max_steps 128 --n_updates 3 --batch_size 32 --n_trials 1 --epoch 5
```

Check the output to ensure everything works before submitting large jobs.

## Quick Reference

### Available Partitions

```bash
sinfo
```

Common partitions on WPI Turing:
- `academic`: 2-day time limit
- `short`: 1-day time limit (default)
- `long`: 7-day time limit
- `quick`: 12-hour time limit

### Useful SLURM Commands

```bash
# Job management
sbatch <script.sh>          # Submit job
squeue -u $USER             # List your jobs
scancel <jobid>             # Cancel job
scontrol show job <jobid>   # Job details

# Cluster info
sinfo                       # Partition/node status
sacct -u $USER             # Job accounting info
```

### Command-Line Arguments

Common arguments:
- `-a, --alg`: Algorithm (PPO, ES, ESPPO, MAXPPO, ALTPPO, CMA_PPO)
- `--n_trials`: Number of trials (default: 5)
- `--epoch`: Max iterations (default: 10000)
- `--ppo_lr`: PPO learning rate
- `--es_lr`: ES learning rate
- `--directory`: Save directory (default: ./checkpoints)

See `CODEBASE_OVERVIEW.md` for full argument documentation.

## Troubleshooting Checklist

Before asking for help, check:

- [ ] Conda environment `myenv` exists and is activated
- [ ] All required packages are installed (torch, numpy, matplotlib, gymnasium)
- [ ] Box2D is installed (`pip install gymnasium[box2d]`)
- [ ] tqdm is installed (`pip install tqdm`)
- [ ] Scripts have execute permissions (`chmod +x runner.sh search.sh`)
- [ ] Partition is available (`sinfo`)
- [ ] Test job runs successfully
- [ ] Output files are being created

## Getting Help

If you encounter issues:

1. Check the error output: `cat slurm-<jobid>.err`
2. Verify your environment setup (Environment Setup section)
3. Test with a minimal job (Quick Test section)
4. Check cluster status and your account limits
5. Review this guide's troubleshooting section

## Additional Resources

- **SLURM Documentation**: https://slurm.schedmd.com/
- **Project Documentation**: See `CODEBASE_OVERVIEW.md` for algorithm details
- **CMA-PPO Documentation**: See `CMA_PPO.md` for CMA-PPO specific details
- **WPI Cluster Documentation**: Check WPI-specific cluster documentation for partition policies and limits

