# ES-PPO vs CMA-PPO: Independent Trial Comparison

## Executive Summary

This document presents a fair comparison between ES-PPO and CMA-PPO algorithms using **independent trials** with matched computational budgets. Each algorithm runs **3 independent trials**, each with **3.84M environment steps** (1.5x the original 2.56M). Trials are truly independent: each starts with a fresh, randomly initialized policy using trial-specific seeds.

## Independent Trial Configuration

### Key Changes from Previous Runs

1. **Trial Independence**: Each trial now starts with a fresh policy (weights reinitialized using PyTorch defaults)
2. **Trial-Specific Seeds**: Each trial uses `base_seed + trial_num` for reproducibility
3. **Separate Jobs**: Each trial runs as an independent Slurm job for true parallelism
4. **Statistical Validity**: Independent trials enable proper statistical analysis (mean, std, confidence intervals)

### ES-PPO Configuration (Per Trial)

- **Algorithm**: ES-PPO hybrid
- **Max steps per iteration**: 256
- **Population size**: 5
- **Batch size**: 32
- **n_updates (epochs)**: 1
- **Total iterations**: 3,000
- **Total computation**: 256 × 5 × 3,000 = **3,840,000 env steps**
- **Final policy steps**: 256 × 3,000 = 768,000 env steps
- **Gradient updates per iteration**: 5 population × 8 minibatches × 1 epoch = **40 gradient steps**
- **Total gradient updates**: 40 × 3,000 = **120,000 gradient steps**

### CMA-PPO Configuration (Per Trial)

- **Algorithm**: CMA-PPO
- **Max steps per iteration**: 4,096
- **Batch size**: 1,024
- **n_updates (epochs)**: 78
- **Total iterations**: 937
- **Total computation**: 4,096 × 937 = **3,837,952 env steps** (0.05% difference from ES-PPO)
- **Final policy steps**: 4,096 × 937 = 3,837,952 env steps
- **Minibatches per epoch**: 4,096 ÷ 1,024 = **4 minibatches**
- **Gradient updates per iteration**: 4 minibatches × 78 epochs × 3 networks = **936 gradient steps**
- **Total gradient updates**: 936 × 937 = **877,032 gradient steps**

**Key Point**: Both configurations use **~3.84M env steps per trial**, making them fair in terms of environment interaction. CMA-PPO does **7.3x more gradient updates** (877k vs 120k) to match ES-PPO's computational overhead.

## Why Independent Trials Matter

### Previous Issue: Sequential Trials

The original implementation had a critical flaw:
- **Same policy object reused** across all trials
- Policy weights **carried over** from trial to trial
- Trials were **not independent** (Trial 2 continued from Trial 1's weights)
- Statistics were **less meaningful** (not true independent samples)

### Current Fix: True Independence

- **Policy reinitialized** at start of each trial using PyTorch defaults
- **Trial-specific seeds** ensure different random weights per trial
- **Separate Slurm jobs** enable true parallelism
- **Proper statistics** can now be computed (mean, std, confidence intervals)

### Implementation Details

**Policy Reinitialization** (in `experiments/runner.py`):
- Uses PyTorch's default Linear initialization: Uniform U(-sqrt(1/in_features), sqrt(1/in_features))
- Applied with trial-specific seed (set before reset)
- All trials use the same initialization scheme (consistent with original `get_policy()`)
- Each trial gets different random weights (due to different seeds)

## Experimental Setup

### Job Submission

- **6 total jobs**: 3 ES-PPO + 3 CMA-PPO
- **All submitted concurrently** for parallel execution
- **Output directory**: `independent_trials_YYYYMMDD_HHMMSS/`
- **Each trial in separate subdirectory**: `es_ppo_trial_N/` or `cma_ppo_trial_N/`

### Trial Configuration

| Trial | ES-PPO Seed | CMA-PPO Seed | Notes |
|-------|-------------|--------------|-------|
| 1 | 1235 | 1235 | base_seed (1234) + 1 |
| 2 | 1236 | 1236 | base_seed (1234) + 2 |
| 3 | 1237 | 1237 | base_seed (1234) + 3 |

### Progress Monitoring

Use `./check_progress.sh` to check all trials:
- Automatically finds latest `independent_trials_*` directory
- Parses progress from each trial's slurm output
- Shows status, progress, rewards, and summary statistics
- Provides deterministic, consistent progress information

## Previous Run Analysis (For Reference)

### Original Sequential Runs

**ES-PPO (Sequential Trials)**:
- Trial 1: -16.75 (final), -7.94 (best)
- Trial 2: 185.29 (final), 196.73 (best)
- Trial 3: 206.59 (final), 228.90 (best)
- Trial 4: 220.72 (final), 230.02 (best)
- Trial 5: 226.66 (final), 235.41 (best)
- **Best overall**: 235.41
- **Average final**: 164.50
- **Runtime**: ~4.05 hours total (sequential)

**CMA-PPO (Sequential Trials)**:
- Trial 1: 144.28 (final), 218.98 (best)
- Trial 2: -117.12 (final), 235.46 (best)
- Trial 3: In progress, negative rewards observed
- **Best overall**: 235.46
- **Average final**: 13.58 (from 2 completed trials)
- **Runtime**: ~5.3 hours (sequential, with 26x compute increase)

**Key Observations**:
- Both algorithms reached similar peak performance (~235)
- ES-PPO showed more consistent improvement across trials
- CMA-PPO Trial 3 showed degradation (likely due to sequential nature)
- Trials were not independent, limiting statistical validity

## Expected Results (New Independent Trials)

### Fair Comparison Criteria

1. **Equal env steps**: ✓ Both use ~3.84M env steps per trial
2. **Independent trials**: ✓ Each trial starts fresh with different seed
3. **Parallel execution**: ✓ All 6 jobs run concurrently
4. **Proper statistics**: ✓ Can compute mean, std, confidence intervals

### What to Analyze

1. **Performance Comparison**:
   - Best reward per algorithm
   - Average reward across trials
   - Standard deviation (consistency)
   - Statistical significance

2. **Convergence Analysis**:
   - Iterations to reach target performance
   - Learning curves per trial
   - Variance in convergence speed

3. **Algorithmic Efficiency**:
   - Performance per env step
   - Performance per gradient update
   - Sample efficiency comparison

## Configuration Summary

| Metric | ES-PPO (Per Trial) | CMA-PPO (Per Trial) |
|--------|-------------------|---------------------|
| Max steps | 256 | 4,096 |
| Population size | 5 | 1 |
| Total iterations | 3,000 | 937 |
| Batch size | 32 | 1,024 |
| n_updates (epochs) | 1 | 78 |
| Total env steps | 3,840,000 | 3,837,952 |
| Gradient steps/iter | 40 | 936 |
| Total gradient steps | 120,000 | 877,032 |
| Number of trials | 3 | 3 |
| Trial independence | ✓ Yes | ✓ Yes |

## Conclusion

The new independent trial setup enables:

1. **True Statistical Comparison**: Independent trials allow proper statistical analysis
2. **Fair Runtime Comparison**: Both algorithms use similar env step budgets with matched compute
3. **Parallel Execution**: All 6 jobs run concurrently, reducing total wall-clock time
4. **Reproducibility**: Trial-specific seeds ensure reproducibility while maintaining independence

Results from these runs will provide a fair, statistically valid comparison between ES-PPO and CMA-PPO algorithms.
