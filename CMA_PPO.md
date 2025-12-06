# CMA-PPO Implementation Documentation

## Overview

This document describes the implementation of **CMA-PPO (Covariance Matrix Adaptation PPO)** following Hämäläinen et al. (2018). CMA-PPO is a variant of Proximal Policy Optimization that uses separate mean and variance networks with action mirroring and history-based updates.

## Algorithm Description

### Key Differences from Vanilla PPO

1. **Separate Networks**: The actor is split into two independent networks:
   - **Mean Network**: Predicts the mean of the action distribution
   - **Variance Network**: Predicts the variance of the action distribution
   - Both networks have separate optimizers

2. **Pre-tanh Actions**: Actions are sampled in pre-tanh space, then tanh-squashed before being sent to the environment. The algorithm works with pre-tanh actions internally.

3. **Action Mirroring**: Negative-advantage actions are mirrored around the current mean with Gaussian kernel weighting:
   - Mirror: `a_mirrored = 2 * μ(s) - a`
   - Weight: `w = exp(-||a - μ(s)||² / (2 * σ²))`

4. **History Buffer**: Maintains a buffer of the last H iterations (default H=5) for rank-µ variance updates.

5. **Update Order**:
   - **Variance Network**: Trained first using minibatches from all history data (rank-µ update)
   - **Mean Network**: Trained using only current iteration's CMA batch
   - **Value Network**: Trained using current iteration's data

6. **Unclipped Objective**: Uses unclipped log-likelihood weighted by advantages:
   - Loss: `-A * log_prob` (no PPO clipping, no entropy bonus)

7. **GAE**: Uses Generalized Advantage Estimation (GAE) with λ parameter (default λ=0.95)

## Implementation Structure

### Files Created/Modified

#### 1. `policies/continuous.py`
- **Added**: `CMAPPOPolicyContinuous` class
  - Separate `mean_net` and `var_net` networks
  - Value function (`vf`) unchanged from PPO
  - `forward()` method returns pre-tanh actions
  - `evaluate()` method works with pre-tanh actions
  - `get_mean_var()` helper method

#### 2. `algorithms/core/cma_ppo_core.py`
- **Added**: `CMA_PPOUpdater` class
  - Implements core CMA-PPO update logic
  - History buffer management
  - Action mirroring with Gaussian kernel
  - GAE computation
  - Rank-µ variance updates
  - Separate optimizers for mean, var, and value networks

#### 3. `algorithms/standalone/cma_ppo.py`
- **Added**: `CMA_PPO` class
  - Complete algorithm implementation
  - Implements `Algorithm` interface
  - Wraps `CMA_PPOUpdater`
  - Handles checkpointing for all three optimizers

#### 4. `envs/wrappers.py`
- **Added**: `run_env_CMA_PPO()` function
  - Environment wrapper for CMA-PPO
  - Returns pre-tanh actions along with tanh-squashed actions
  - Compatible with CMA-PPO's requirements

#### 5. `experiments/registry.py`
- **Modified**: Added CMA-PPO to algorithm registry
  - Registered as `'cma_ppo'` and `'cmappo'`

#### 6. `options.py`
- **Modified**: Added CMA-PPO specific arguments:
  - `--lam`: GAE lambda parameter (default: 0.95)
  - `--cma_lr_mean`: Mean network learning rate (default: 3e-4)
  - `--cma_lr_var`: Variance network learning rate (default: 3e-4)
  - `--cma_lr_value`: Value network learning rate (default: 1e-3)
  - `--history_size`: History buffer size H (default: 5)
  - `--kernel_std`: Gaussian kernel std for mirroring (default: 0.1)

#### 7. `main.py`
- **Modified**: Added CMA-PPO configuration handling

## Architecture Details

### Policy Network Structure

```python
CMAPPOPolicyContinuous:
  ├── mean_net: Sequential(
  │     Linear(24 → 100) → ReLU → 
  │     Linear(100 → 100) → Tanh → ReLU → 
  │     Linear(100 → 4)
  │   )
  ├── var_net: Sequential(
  │     Linear(24 → 100) → ReLU → 
  │     Linear(100 → 100) → ReLU → 
  │     Linear(100 → 4)
  │   )
  └── vf: Sequential(
        Linear(24 → 100) → ReLU → 
        Linear(100 → 100) → ReLU → 
        Linear(100 → 1)
      )
```

### Update Flow

1. **Collect Trajectories**: Run environment, collect states, actions (pre-tanh and tanh), rewards, values
2. **Compute GAE**: Calculate advantages using GAE with λ parameter
3. **Build CMA Batch**: 
   - Include all positive-advantage samples
   - Mirror negative-advantage actions around current mean
   - Weight mirrored actions with Gaussian kernel
4. **Update Variance Network**:
   - Sample minibatches from history buffer (last H iterations)
   - Build CMA batches for each history entry
   - Update using: `loss = -A * log_prob`
5. **Update Mean Network**:
   - Use only current iteration's CMA batch
   - Update using: `loss = -A * log_prob`
6. **Update Value Network**:
   - Use current iteration's data
   - Update using: `loss = MSE(value_pred, returns)`

### History Buffer

The history buffer stores the last H iterations (default H=5) of:
- States
- Pre-tanh actions
- Advantages
- Returns

This allows the variance network to learn from a larger, more diverse dataset (rank-µ update).

### Action Mirroring

For actions with negative advantages:
1. Compute mirrored action: `a_mirrored = 2 * μ(s) - a`
2. Compute Gaussian kernel weight: `w = exp(-||a - μ(s)||² / (2 * σ²))`
3. Weight by absolute advantage: `w_total = w * |A|`
4. Sample mirrored actions according to weights
5. Use positive advantages for mirrored actions (they're now "good")

## Hyperparameters

### Default Values

- `gamma`: 0.99 (discount factor)
- `lam`: 0.95 (GAE lambda)
- `lr_mean`: 3e-4 (mean network learning rate)
- `lr_var`: 3e-4 (variance network learning rate)
- `lr_value`: 1e-3 (value network learning rate)
- `batch_size`: 2048 (minibatch size)
- `max_steps`: 16000 (steps per iteration)
- `n_updates`: 5 (number of update epochs)
- `history_size`: 5 (H, history buffer size)
- `kernel_std`: 0.1 (Gaussian kernel std for mirroring)

### Recommended Ranges

From the paper specification:
- `gamma`: 0.99 (fixed)
- `lam`: 0.95 (GAE lambda)
- `lr_mean`: 3e-4 (among {0.0001, 0.00025, 0.001})
- `lr_var`: 3e-4 (same as mean)
- `lr_value`: 1e-3 (among {0.0001, 0.0025, 0.001})
- `batch_size`: ~2048 (for steps_per_iter≈16k)
- `n_updates`: 5 (epochs for value/mean/variance)

## Usage

### Basic Usage

```bash
python main.py -a CMA_PPO \
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
```

### Quick Test

```bash
python main.py -a CMA_PPO \
    --n_updates 3 \
    --batch_size 512 \
    --max_steps 1000 \
    --epoch 100 \
    --n_trials 1
```

### SLURM Submission

```bash
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
```

## Integration with Existing Framework

CMA-PPO integrates seamlessly with the existing modular framework:

- ✅ Uses same `Algorithm` interface
- ✅ Works with `ExperimentRunner` for fair comparisons
- ✅ Compatible with `MetricsTracker`
- ✅ Supports checkpointing and loading
- ✅ Follows same evaluation protocol
- ✅ Real-time progress tracking with tqdm
- ✅ Periodic plot generation

## Metrics

CMA-PPO returns the following metrics:

- `cma_ppo/mean_loss`: Mean network loss
- `cma_ppo/var_loss`: Variance network loss
- `cma_ppo/value_loss`: Value network loss
- `cma_ppo/advantage_mean`: Mean of advantages
- `cma_ppo/advantage_std`: Std of advantages
- `episode_reward`: Evaluation reward
- `policy/weight_l2_norm`: Policy weight L2 norm
- `policy/weight_l2_distance_from_init`: Weight drift from initialization
- `policy/param_mean`: Parameter mean
- `policy/param_std`: Parameter std

## Differences from Paper Implementation

1. **Network Architecture**: We use a simpler architecture (100-100-4) compared to potentially larger networks in the paper. This can be adjusted.

2. **History Buffer**: We maintain a fixed-size deque (H=5), which automatically discards old data.

3. **Mirroring Sampling**: We sample mirrored actions according to weights, balancing positive and mirrored samples.

4. **Value Network**: Uses standard MSE loss, same as PPO.

## Testing Status

✅ **Verified Working** (December 6, 2024)
- Successfully ran test experiment with 50 iterations
- All components functioning correctly:
  - Policy forward pass with pre-tanh actions
  - Environment interaction
  - GAE computation
  - Action mirroring
  - History buffer updates
  - Separate network updates (mean, var, value)
- Results saved successfully
- Progress tracking working (tqdm, real-time rewards)
- No runtime errors

Test command used:
```bash
python main.py -a CMA_PPO --n_updates 3 --batch_size 512 --max_steps 1000 --epoch 50 --n_trials 1
```

## Implementation Notes

- The implementation works in pre-tanh space, which is important for the mirroring operation
- The history buffer allows the variance network to learn from more diverse data
- Action mirroring helps convert "bad" actions into "good" ones by reflecting them around the mean
- The separate optimizers allow independent learning rates for mean, variance, and value networks
- All three networks (mean, var, value) are updated in separate optimization steps

## References

Hämäläinen, P., Babadi, A., Ma, X., & Jegelka, S. (2018). PPO-CMA: Proximal Policy Optimization with a Covariance Matrix Adaptation Update Rule. *International Conference on Learning Representations (ICLR)*.

