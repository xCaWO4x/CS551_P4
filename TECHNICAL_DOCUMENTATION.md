# Technical Documentation

Complete technical documentation for the CS551_P4 modular ES+PPO hybrid algorithms framework.

## Table of Contents

1. [Codebase Overview](#codebase-overview)
2. [CMA-PPO Implementation](#cma-ppo-implementation)
3. [Adaptive History Scheduling](#adaptive-history-scheduling)
4. [Goal Detection and Stabilization](#goal-detection-and-stabilization)
5. [Running Experiments](#running-experiments)

---

## Codebase Overview

### Project Goals

1. **Easy Algorithm Composition**: Plug in new hybrid algorithms without copy-pasting PPO/ES logic
2. **Fair Comparisons**: Same env, policy arch, seeds, logging across all experiments
3. **Deep Inspection**: Track PPO stats (KL, clip fraction), ES stats (variance, gradient norms), policy drift

### Project Structure

```
CS551_P4/
├── algorithms/
│   ├── base.py              # Algorithm interface
│   ├── core/                # Shared components
│   │   ├── ppo_core.py      # Reusable PPO update logic
│   │   ├── es_core.py       # Reusable ES update logic
│   │   └── cma_ppo_core.py  # CMA-PPO update logic
│   ├── standalone/         # Standalone algorithms
│   │   ├── ppo.py
│   │   ├── es.py
│   │   └── cma_ppo.py
│   └── hybrid/              # Hybrid algorithms
│       ├── es_ppo.py
│       ├── max_ppo.py
│       └── alt_ppo.py
│
├── experiments/
│   ├── runner.py            # Experiment runner
│   └── registry.py          # Algorithm registry
│
├── metrics/
│   └── tracker.py           # Metrics collection
│
├── policies/
│   └── continuous.py       # Policy networks
│
├── envs/
│   └── wrappers.py         # Environment utilities
│
├── utils/
│   └── torch_utils.py      # Tensor utilities
│
├── main.py                 # Entry point
├── options.py              # Command-line argument parsing
├── runner.sh               # SLURM job template
└── plot.py                 # Plotting utilities
```

### Key Components

#### 1. Algorithm Interface (`algorithms/base.py`)

All algorithms implement a standard interface:
- `step()` - Perform one training step, returns metrics dict
- `get_policy()` - Return current policy
- `save_checkpoint()` / `load_checkpoint()` - State management
- `model_name` - Unique identifier

#### 2. Shared Core Components

**`algorithms/core/ppo_core.py`** - `PPOUpdater` class
- Reusable PPO update logic
- Returns metrics: KL divergence, clip fraction, value loss, policy loss, entropy, advantage stats
- Can be used by standalone PPO or hybrid algorithms

**`algorithms/core/es_core.py`** - `ESUpdater` class
- Reusable ES logic for population generation and weight updates
- Returns metrics: reward variance, gradient estimate norm, population stats
- Handles parallel evaluation via ThreadPool

**`algorithms/core/cma_ppo_core.py`** - `CMA_PPOUpdater` class
- CMA-PPO update logic with action mirroring and history buffer
- Returns metrics: mean loss, var loss, value loss, advantage stats
- See [CMA-PPO Implementation](#cma-ppo-implementation) for details

#### 3. Experiment System

**`experiments/runner.py`** - `ExperimentRunner` class
- Ensures fair comparison (same seeds, policy initialization, evaluation)
- Manages trial execution
- Coordinates metrics collection
- Saves results and generates plots
- Real-time progress tracking with tqdm
- Periodic plot generation (every 50 evaluations)
- Goal detection and adaptive history integration

**`experiments/registry.py`** - Algorithm registry
- Maps algorithm names to classes
- Easy to add new algorithms
- Supports multiple aliases (e.g., `'cma_ppo'` and `'cmappo'`)

#### 4. Metrics System

**`metrics/tracker.py`** - `MetricsTracker` class
- Centralized metrics collection
- Tracks PPO metrics (KL div, clip fraction, losses)
- Tracks ES metrics (variance, gradient norms)
- Tracks CMA-PPO metrics (mean/var/value losses)
- Tracks policy drift (weight distance from initialization)
- Saves metrics to disk

### Algorithm Composition Pattern

#### Standalone Algorithms

```python
class PPO(Algorithm):
    def __init__(self, ...):
        self.ppo_updater = PPOUpdater(policy, config)
    
    def step(self):
        metrics = self.ppo_updater.update(n_sequences=1)
        reward = evaluate()
        return {**metrics, 'episode_reward': reward}
```

#### Hybrid Algorithms

```python
class ESPPO(Algorithm):
    def __init__(self, ...):
        self.ppo_updater = PPOUpdater(policy, config)
        self.es_updater = ESUpdater(policy, config)
    
    def step(self):
        # 1. Generate ES population
        population = self.es_updater.generate_population()
        
        # 2. Run PPO on each
        results = []
        for weights in population:
            set_weights(weights)
            ppo_metrics = self.ppo_updater.update(...)
            reward = evaluate()
            results.append((weights, reward, ppo_metrics))
        
        # 3. ES update based on results
        es_metrics = self.es_updater.update(rewards, epsilons)
        
        # 4. Return combined metrics
        return {**ppo_metrics, **es_metrics, 'episode_reward': ...}
```

### Fair Comparison Guarantees

The `ExperimentRunner` ensures:
1. **Same Environment**: All algorithms use identical environment instances
2. **Same Policy Architecture**: All use same network structure
3. **Same Initialization**: Policies initialized with same seed
4. **Same Seeds**: All randomness controlled via config
5. **Same Evaluation**: Evaluation happens at same intervals with same conditions
6. **Same Logging**: All metrics collected in same format

---

## CMA-PPO Implementation

### Overview

**CMA-PPO (Covariance Matrix Adaptation PPO)** follows Hämäläinen et al. (2018). CMA-PPO is a variant of Proximal Policy Optimization that uses separate mean and variance networks with action mirroring and history-based updates.

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

### Architecture

#### Policy Network Structure

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

#### Update Flow

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

### Hyperparameters

#### Default Values

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

### Usage

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

### Metrics

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

### References

Hämäläinen, P., Babadi, A., Ma, X., & Jegelka, S. (2018). PPO-CMA: Proximal Policy Optimization with a Covariance Matrix Adaptation Update Rule. *International Conference on Learning Representations (ICLR)*.

---

## Adaptive History Scheduling

### Overview

The adaptive history-length scheduling system dynamically adjusts the history buffer size (H) based on training performance, enabling more exploration during early training and stabilization once the goal is reached.

### Motivation

CMA-PPO uses a history buffer to store the last H iterations of training data for variance network updates (rank-µ updates). The history length H is a critical hyperparameter:

- **Longer history (H_max)**: More exploration, robust CMA evolution path, better for early training
- **Shorter history (H_min)**: Less aggressive exploration, stabilizes gait, better for fine-tuning

The adaptive scheduling system automatically transitions from longer to shorter history as performance improves.

### Algorithm

The history length H is adjusted based on the best evaluation reward seen so far (`R_best`):

1. **Pre-goal phase** (`R_best ≤ R_goal`):
   - `H = H_max`
   - Longer history for robust exploration

2. **Post-goal phase** (`R_best ≥ R_high`):
   - `H = H_min`
   - Shorter history for stabilization

3. **Transition phase** (`R_goal < R_best < R_high`):
   - Linear interpolation between `H_max` and `H_min`
   - `α = (R_high - R_best) / (R_high - R_goal)`
   - `H = ceil(H_min + (H_max - H_min) * α)`

### Implementation

The system is integrated into `CMA_PPO` and automatically called after each evaluation:

```python
def update_history_config(self, eval_reward: float, best_eval_reward: Optional[float] = None):
    # Update best reward tracker
    self.best_eval_reward = max(self.best_eval_reward, best_eval_reward or eval_reward)
    
    # Compute new H based on R_best
    if R_best <= R_goal:
        new_H = H_max
    elif R_best >= R_high:
        new_H = H_min
    else:
        # Linear interpolation
        alpha = (R_high - R_best) / (R_high - R_goal)
        new_H = ceil(H_min + (H_max - H_min) * alpha)
    
    # Resize history buffer if changed
    if new_H != self.H:
        self.H = new_H
        self.cma_ppo_updater.set_history_size(new_H)
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `history_size` | 5 | Maximum history length (H_max) |
| `history_len_min` | 1 | Minimum history length (H_min) |
| `reward_goal` | None | Target reward threshold (from runner) |
| `reward_high` | `goal + 50` | High confidence threshold |

### Usage

```bash
python main.py \
    --alg CMA_PPO \
    --history_size 5 \
    --history_len_min 1 \
    --reward_high 300.0 \
    --epoch 1000
```

The system is fully automatic once `reward_goal` is set in the ExperimentRunner.

---

## Goal Detection and Stabilization

### Overview

The goal detection and stabilization system detects when a policy consistently achieves a predefined reward threshold and automatically saves policy checkpoints. This addresses the issue of policies that reach high rewards but may not be stable.

### Motivation

During training, algorithms like CMA-PPO may experience early spikes in performance where the policy temporarily reaches high rewards. However, these spikes may not be stable:

1. **Noisy Evaluations**: Single evaluation may hit a high reward due to stochasticity
2. **Temporary Performance**: Policy may briefly reach goal but then degrade
3. **Inconsistent Behavior**: Policy may work well in one evaluation but fail in others

The goal detection system addresses these issues by:
- **Smoothing**: Using a moving window average to reduce noise
- **Confirmation**: Requiring multiple consecutive evaluations in the goal band
- **Policy Snapshotting**: Saving the policy state when goal is first confirmed
- **Early Stopping**: Optionally stopping training when goal is reached

### Algorithm

For each evaluation:

1. **Update Best Reward**: Update `best_eval_reward` if current reward is greater
2. **Update History**: Append current evaluation reward to `eval_history`
3. **Compute Smoothed Reward**: Calculate moving average over last `goal_window` evaluations
4. **Check Goal Band**: Determine if smoothed reward OR current reward is in goal band:
   ```
   threshold = reward_goal - goal_delta
   in_goal_band = (barE >= threshold) OR (current_reward >= threshold)
   ```
5. **Update Consecutive Counter**:
   - If in goal band: `goal_consecutive += 1`
   - Else: `goal_consecutive = 0`
6. **Confirm Goal** (first time only):
   - If `goal_consecutive >= goal_min_consecutive` and `steps_to_goal is None`:
     - Record `steps_to_goal = total_env_steps`
     - Save full checkpoint via `algorithm.save_checkpoint()` to `trial_N_goal_policy.pkl`
     - Log goal detection event
     - If `early_stop_on_goal=True`: raise `StopIteration` to stop training

### Configuration Parameters

- **`goal_delta`** (default: 15.0): Allowed noise below the goal threshold. The goal band is defined as `[goal - goal_delta, ∞)`.
- **`goal_window`** (default: 3): Size of the smoothing window W. The system computes a moving average of the last W evaluations.
- **`goal_min_consecutive`** (default: 2): Minimum number of consecutive evaluations that must be in the goal band before confirming goal detection.
- **`early_stop_on_goal`** (default: False): If True, training stops immediately when goal is confirmed.

### Usage

```bash
python main.py \
    --alg CMA_PPO \
    --epoch 1000 \
    --goal_delta 50.0 \
    --goal_window 3 \
    --goal_min_consecutive 2 \
    --early_stop_on_goal
```

### Output Files

When goal is detected, the system saves:
- **`trial_N_goal_policy.pkl`**: Policy checkpoint at the moment goal was first confirmed
  - Contains: `policy_state_dict`, `optimizer_mean_state_dict`, `optimizer_var_state_dict`, `optimizer_value_state_dict`

### Example Scenarios

**Early Spike Detection**:
```
Iteration 50: reward = 180.0
Iteration 60: reward = 245.0  (in goal band, consecutive=1)
Iteration 70: reward = 252.0  (in goal band, consecutive=2) → GOAL CONFIRMED
  → Policy saved to trial_1_goal_policy.pkl
  → steps_to_goal = 286720
```

**Noisy Evaluations**:
```
Iteration 50: reward = 180.0
Iteration 60: reward = 255.0  (in goal band, consecutive=1)
Iteration 70: reward = 190.0  (below goal band, consecutive=0)
Iteration 80: reward = 248.0  (in goal band, consecutive=1)
Iteration 90: reward = 251.0  (in goal band, consecutive=2) → GOAL CONFIRMED
```

The system correctly ignores the single spike at iteration 60 and confirms only when stability is achieved.

---

## Running Experiments

### Environment Setup

#### 1. Create/Activate Conda Environment

```bash
# Create the environment (Python 3.11 recommended)
conda create -n myenv python=3.11 -y

# Activate the environment
conda activate myenv

# Install required packages
pip install torch numpy matplotlib gymnasium
pip install gymnasium[box2d]  # Required for BipedalWalker
pip install tqdm  # Optional, for progress bars
```

#### 2. Verify Installation

```bash
conda activate myenv
python -c "import torch; import numpy; import matplotlib; import gymnasium; env = gymnasium.make('BipedalWalker-v3'); print('Setup complete!')"
```

### SLURM Scripts

The project includes several SLURM scripts:

- **`runner.sh`**: Template script for submitting individual jobs
- **`search.sh`**: Script that submits multiple jobs for hyperparameter search
- **`run_baseline.sh`**: Script to run baseline experiments with paper configurations
- **`submit_cma_ppo_adaptive_trials.sh`**: Script to submit multiple CMA-PPO trials with adaptive history

### Running Single Experiments

#### Basic Command

```bash
sbatch runner.sh python main.py -a PPO --ppo_lr 0.0001 --max_steps 256 --n_updates 5 --batch_size 64 --gamma 0.99 --clip 0.01 --ent_coeff 0.0 --n_trials 3 --epoch 100
```

#### Examples for Different Algorithms

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
```

### Baseline Configurations

#### PPO Baseline

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

#### ES-PPO Baseline

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

### Monitoring Jobs

#### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View specific job
squeue -j <jobid>

# View job details
scontrol show job <jobid>
```

#### Viewing Output

Job output is saved to:
- **Standard output**: `slurm-<jobid>.out`
- **Standard error**: `slurm-<jobid>.err`

View output in real-time (if job is running):
```bash
tail -f slurm-<jobid>.out
```

### Common Issues and Solutions

#### Issue 1: "EnvironmentNameNotFound: Could not find conda environment: pytorch"

**Solution**: The scripts use `myenv`, not `pytorch`. Make sure you've created the `myenv` environment.

#### Issue 2: "ModuleNotFoundError: No module named 'Box2D'"

**Solution**: Install Box2D in your conda environment:
```bash
conda activate myenv
pip install gymnasium[box2d]
```

#### Issue 3: "Invalid account or account/partition combination specified"

**Solution**: Check available partitions:
```bash
sinfo
```

Then edit `runner.sh` to use an available partition (e.g., change `academic` to `short` or `long`).

### Best Practices

1. **Start small**: Test with `--n_trials 1 --epoch 10` before running full experiments
2. **Monitor resources**: Use `squeue` and `sinfo` to check cluster status
3. **Clean up**: Remove old output files periodically to save disk space
4. **Use descriptive job names**: Edit `--job-name` in `runner.sh` for different experiments

---

## Adding a New Algorithm

### Example: Adaptive Sigma ESPPO

**File**: `algorithms/hybrid/adaptive_sigma_esppo.py`

```python
from algorithms.core.ppo_core import PPOUpdater
from algorithms.core.es_core import ESUpdater
from algorithms.base import Algorithm

class AdaptiveSigmaESPPO(Algorithm):
    """ES-PPO that adapts sigma based on reward variance."""
    
    def __init__(self, policy, env_func, config, metrics_tracker=None):
        self.ppo_updater = PPOUpdater(policy, env_func, **config['ppo'])
        self.es_updater = ESUpdater(policy, env_func, **config['es'])
        self.variance_threshold = config.get('variance_threshold', 10.0)
    
    def step(self):
        # Generate population
        population = self.es_updater.generate_population()
        
        # Run PPO on each
        results = []
        for weights in population:
            set_weights(weights)
            ppo_metrics = self.ppo_updater.update()
            reward = evaluate()
            results.append((weights, reward, ppo_metrics))
        
        # Adapt sigma based on variance
        rewards = [r[1] for r in results]
        variance = np.var(rewards)
        if variance < self.variance_threshold:
            self.es_updater.sigma *= 1.1
        else:
            self.es_updater.sigma *= 0.9
        
        # ES update
        es_metrics = self.es_updater.update(rewards, epsilons)
        
        return {**ppo_metrics, **es_metrics, 'episode_reward': ...}
```

**Register**: Add to `experiments/registry.py`

```python
from algorithms.hybrid.adaptive_sigma_esppo import AdaptiveSigmaESPPO

ALGORITHM_REGISTRY['adaptive_sigma_esppo'] = AdaptiveSigmaESPPO
```

**Total**: ~50-100 lines, zero duplication!

---

## Metrics Collected

All algorithms return standardized metrics:

```python
{
    # Training metrics
    'episode_reward': float,
    
    # PPO metrics (if applicable)
    'ppo/kl_divergence': float,
    'ppo/clip_fraction': float,
    'ppo/policy_loss': float,
    'ppo/value_loss': float,
    'ppo/entropy': float,
    'ppo/advantage_mean': float,
    'ppo/advantage_std': float,
    'ppo/ratio_mean': float,
    'ppo/ratio_std': float,
    
    # ES metrics (if applicable)
    'es/reward_variance': float,
    'es/gradient_norm_mean': float,
    'es/population_mean_reward': float,
    'es/population_std_reward': float,
    'es/sigma': float,
    
    # CMA-PPO metrics (if applicable)
    'cma_ppo/mean_loss': float,
    'cma_ppo/var_loss': float,
    'cma_ppo/value_loss': float,
    'cma_ppo/advantage_mean': float,
    'cma_ppo/advantage_std': float,
    
    # Policy metrics (all algorithms)
    'policy/weight_l2_norm': float,
    'policy/weight_l2_distance_from_init': float,
    'policy/param_mean': float,
    'policy/param_std': float,
}
```

---

## References

- Hämäläinen, P., Babadi, A., Ma, X., & Jegelka, S. (2018). PPO-CMA: Proximal Policy Optimization with a Covariance Matrix Adaptation Update Rule. *International Conference on Learning Representations (ICLR)*.

