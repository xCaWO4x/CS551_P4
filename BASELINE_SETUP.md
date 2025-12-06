# Baseline Setup Documentation

This document describes the baseline experiments setup for CS551 Project 4, including the configuration, bug fixes, and workflow.

## Overview

We established baseline experiments for **PPO** and **ES-PPO** algorithms using the exact hyperparameters from the original paper. These baselines serve as the reference point for comparing future improvements (e.g., CMA-ES variants).

## Baseline Configuration

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

## Running Baselines

### Quick Start

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

## Expected Runtime

Based on test runs:

- **PPO**: ~2-4 hours (5 trials × 10,000 iterations)
- **ES-PPO**: ~5-10 hours (5 trials × 10,000 iterations × population_size=5)

Note: Times may vary based on:
- Early stopping if reward goal (300) is reached
- Cluster load and GPU availability
- Specific hyperparameter configuration

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f slurm-<jobid>.out

# Check for errors
cat slurm-<jobid>.err
```

## Results Location

Results are saved in `./checkpoints/` with algorithm-specific directory names:

- PPO: `./checkpoints/PPO_3_32_256_0.99_0.2_0.0_0.0001/`
- ES-PPO: `./checkpoints/ESPPO_5_0.1_1_32_256_0.99_0.2_0.0_0.001_0.0001_1/`

Each directory contains:
- `results.txt`: Summary with average final reward and time
- `rewards.txt`: Reward history
- `rewards_plot.png`: Training curve plot
- `weights.pkl`: Final policy weights
- `metrics.pkl`: Detailed metrics (KL divergence, clip fraction, etc.)

## Bug Fixes Applied

### 1. PPO Core Parameter Conflict

**Issue**: `ppo_core.py` was passing `policy=self.policy` as a keyword argument to `env_func`, but the lambda in `ppo.py` already captured `policy`, causing a `TypeError: got multiple values for keyword argument 'policy'`.

**Fix**: Removed the redundant `policy` parameter from the call in `algorithms/core/ppo_core.py` line 64-67. The lambda already captures the policy from closure.

**File**: `algorithms/core/ppo_core.py`

### 2. Search Script Parameter Swap

**Issue**: In `search.sh`, the `gamma` and `clip` parameters were swapped in the command line arguments.

**Fix**: Corrected line 12 to use `--gamma=$n --clip=$m` instead of `--gamma=$m --clip=$n`.

**File**: `search.sh` (now replaced with unified search script)

## Codebase Cleanup

The following cleanup was performed:

1. **Removed old SLURM output files**: Deleted all `slurm-*.out`, `slurm-*.err`, `test_slurm-*.out`, and `test_slurm-*.err` files from previous test runs
2. **Created baseline script**: `run_baseline.sh` for easy baseline execution
3. **Created unified search script**: `search.sh` for future hyperparameter searches (PPO and ESPPO)

## Next Steps

After baseline experiments complete:

1. **Analyze Results**: Review `results.txt` files to verify baselines are working correctly
2. **Compare Performance**: Compare PPO vs ES-PPO performance
3. **Implement Improvements**: Add CMA-ES variants and other enhancements
4. **Compare Against Baselines**: Use these baselines as reference for evaluating improvements

## Files Created/Modified

- `run_baseline.sh`: Script to run baseline experiments
- `search.sh`: Unified hyperparameter search script (for future use)
- `test_defaults.sh`: Quick test script for verification
- `algorithms/core/ppo_core.py`: Fixed parameter conflict bug
- `BASELINE_SETUP.md`: This documentation file

## References

- Original paper hyperparameters from Section 4.1.1 (PPO) and 4.1.3 (ES-PPO)
- Environment: BipedalWalker-v3 (reward goal: 300)
- Framework: Modular ES+PPO hybrid algorithms framework

