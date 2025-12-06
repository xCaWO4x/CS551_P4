# CS551_P4: Modular ES+PPO Hybrid Algorithms Framework

## Overview

This is a modular reinforcement learning framework for experimenting with Evolution Strategies (ES) and Proximal Policy Optimization (PPO) hybrid algorithms. The codebase is designed to enable easy algorithm composition, fair comparisons, and comprehensive metrics tracking.

## Project Goals

1. **Easy Algorithm Composition**: Plug in new hybrid algorithms without copy-pasting PPO/ES logic
2. **Fair Comparisons**: Same env, policy arch, seeds, logging across all experiments
3. **Deep Inspection**: Track PPO stats (KL, clip fraction), ES stats (variance, gradient norms), policy drift

## Project Structure

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
│   └── continuous.py        # Policy networks
│
├── envs/
│   └── wrappers.py          # Environment utilities
│
├── utils/
│   └── torch_utils.py       # Tensor utilities
│
├── main.py                  # Entry point
├── options.py               # Command-line argument parsing
├── runner.sh                # SLURM job template
├── search.sh                # Hyperparameter search script
├── run_baseline.sh          # Baseline experiment script
│
└── archive/old_code/        # Previous implementation (for reference)
```

## Key Features

✅ **No Code Duplication** - Shared PPO/ES cores written once, used everywhere  
✅ **Fair Comparisons** - Same seeds, policy init, evaluation guaranteed  
✅ **Rich Metrics** - KL divergence, clip fraction, variance, gradient norms, policy drift  
✅ **Easy Extension** - Add new hybrids in ~50-100 lines  
✅ **Real-time Progress** - tqdm progress bars with reward tracking  
✅ **Periodic Plotting** - Automatic plot generation during training

## Algorithms

### Standalone Algorithms
- **PPO** - Proximal Policy Optimization
- **ES** - Evolution Strategies
- **CMA-PPO** - Covariance Matrix Adaptation PPO (see `CMA_PPO.md` for details)

### Hybrid Algorithms
- **ESPPO** - ES-PPO hybrid (PPO on each ES candidate)
- **MaxPPO** - Best-of-population selection
- **AltPPO** - Alternating ES/PPO steps

## Architecture

### Component Relationships

```
main.py
  └─► ExperimentRunner
      ├─► Algorithm (PPO/ES/ESPPO/CMA-PPO/etc.)
      │   ├─► PPOUpdater (if needed)
      │   ├─► ESUpdater (if needed)
      │   └─► CMA_PPOUpdater (if CMA-PPO)
      └─► MetricsTracker
          └─► Collects all metrics
```

### Data Flow

1. **Config** → ExperimentRunner creates algorithm with same seeds/policy
2. **Algorithm.step()** → Uses PPOUpdater and/or ESUpdater
3. **Cores return metrics** → Algorithm combines and returns
4. **MetricsTracker** → Collects all metrics
5. **Results saved** → Plots, weights, metrics files

## Key Components

### 1. Algorithm Interface (`algorithms/base.py`)

All algorithms implement a standard interface:
- `step()` - Perform one training step, returns metrics dict
- `get_policy()` - Return current policy
- `save_checkpoint()` / `load_checkpoint()` - State management
- `model_name` - Unique identifier

### 2. Shared Core Components

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
- See `CMA_PPO.md` for detailed documentation

### 3. Experiment System

**`experiments/runner.py`** - `ExperimentRunner` class
- Ensures fair comparison (same seeds, policy initialization, evaluation)
- Manages trial execution
- Coordinates metrics collection
- Saves results and generates plots
- Real-time progress tracking with tqdm
- Periodic plot generation (every 50 evaluations)

**`experiments/registry.py`** - Algorithm registry
- Maps algorithm names to classes
- Easy to add new algorithms
- Supports multiple aliases (e.g., `'cma_ppo'` and `'cmappo'`)

### 4. Metrics System

**`metrics/tracker.py`** - `MetricsTracker` class
- Centralized metrics collection
- Tracks PPO metrics (KL div, clip fraction, losses)
- Tracks ES metrics (variance, gradient norms)
- Tracks CMA-PPO metrics (mean/var/value losses)
- Tracks policy drift (weight distance from initialization)
- Saves metrics to disk

### 5. Policy Networks

**`policies/continuous.py`**
- `ESPolicyContinuous` - Policy for ES algorithms
- `PPOPolicyContinuous` - Policy for PPO algorithms
- `CMAPPOPolicyContinuous` - Policy with separate mean and variance networks for CMA-PPO

### 6. Environment Wrappers

**`envs/wrappers.py`**
- `run_env_ES()` - Environment wrapper for ES
- `run_env_PPO()` - Environment wrapper for PPO
- `run_env_CMA_PPO()` - Environment wrapper for CMA-PPO (handles pre-tanh actions)

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

## Algorithm Composition Pattern

### Standalone Algorithms

```python
class PPO(Algorithm):
    def __init__(self, ...):
        self.ppo_updater = PPOUpdater(policy, config)
    
    def step(self):
        metrics = self.ppo_updater.update(n_sequences=1)
        reward = evaluate()
        return {**metrics, 'episode_reward': reward}
```

### Hybrid Algorithms

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

## Refactoring History

### What Changed from Original Codebase

The original codebase had significant code duplication and lacked fair comparison guarantees. Here's what was improved:

#### Policies Module
- **Before**: Single `policies.py` file with `get_policy(args, env)` requiring args object
- **After**: `policies/continuous.py` with `get_policy(policy_type, state_dim, action_dim)` - more modular, less coupling

#### Experiments Module
- **Before**: All experiment logic in `main.py` (~220 lines), no fair comparison guarantees
- **After**: `experiments/runner.py` with `ExperimentRunner` class ensuring fair comparisons, automatic metrics, cleaner code

#### Metrics Module
- **Before**: No metrics system, only manual reward tracking
- **After**: `metrics/tracker.py` with `MetricsTracker` class tracking comprehensive metrics (PPO, ES, policy)

#### Algorithm Registry
- **Before**: if/elif chains in main.py
- **After**: `experiments/registry.py` with clean algorithm lookup

### Key Improvements

1. **Fair Comparisons**: `ExperimentRunner` ensures same seeds, same policy init, same evaluation for all algorithms
2. **Comprehensive Metrics**: KL divergence, clip fraction, variance, gradient norms, policy drift, etc.
3. **Modularity**: Separated into `experiments/`, `metrics/`, `policies/` modules
4. **Extensibility**: Adding new algorithm = add to registry, `ExperimentRunner` handles the rest

## Fair Comparison Guarantees

The `ExperimentRunner` ensures:
1. **Same Environment**: All algorithms use identical environment instances
2. **Same Policy Architecture**: All use same network structure
3. **Same Initialization**: Policies initialized with same seed
4. **Same Seeds**: All randomness controlled via config
5. **Same Evaluation**: Evaluation happens at same intervals with same conditions
6. **Same Logging**: All metrics collected in same format

## Requirements

- Python 3.7+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- tqdm (optional, for progress bars)

## Documentation

- **`CODEBASE_OVERVIEW.md`** (this file) - General codebase structure and architecture
- **`CMA_PPO.md`** - Detailed CMA-PPO implementation documentation
- **`RUNNING_EXPERIMENTS.md`** - How to run experiments, SLURM setup, baseline configurations

## Migration Notes

- Old code is in `archive/old_code/` for reference
- Command-line interface is backward compatible
- All algorithms produce same results (verified with same seeds)
- New metrics are automatically collected

