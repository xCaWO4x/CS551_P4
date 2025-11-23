# Changes Summary: What Was Used vs What We Changed

## Overview

This document compares the original codebase with the refactored version, focusing on `policies/`, `experiments/`, and `metrics/`.

---

## POLICIES

### Original (`archive/old_code/policies.py`)

**What she had:**
- Single file `policies.py` with:
  - `ESPolicyContinuous` class
  - `PPOPolicyContinuous` class  
  - `get_policy(args, env)` function that:
    - Took `args` object and `env` object
    - Checked `args.alg == 'ES'` to decide which policy
    - Hardcoded `state_dim=24, action_dim=4`
    - Moved policy to CUDA if available

**Key characteristics:**
- Policy selection based on algorithm type
- Required `args` object with `alg` attribute
- Required `env` object (though not really used)
- Hardcoded dimensions for BipedalWalker

### What We Changed (`policies/continuous.py`)

**What we have now:**
- Moved to `policies/continuous.py` (same classes)
- `get_policy(policy_type, state_dim=24, action_dim=4)` function:
  - Takes string `policy_type` instead of `args.alg`
  - Takes explicit `state_dim` and `action_dim` parameters
  - No longer requires `args` or `env` objects
  - Same CUDA logic

**Changes:**
1. ✅ **Moved to module structure** (`policies/` folder)
2. ✅ **Simplified function signature** - removed dependency on `args` object
3. ✅ **More explicit parameters** - `state_dim` and `action_dim` are now parameters
4. ✅ **Same policy classes** - `ESPolicyContinuous` and `PPOPolicyContinuous` unchanged

**Why:**
- Better modularity (policies in their own folder)
- Less coupling (doesn't need `args` object)
- More flexible (can specify dimensions explicitly)

---

## EXPERIMENTS

### Original (in `main.py`)

**What she had:**
- **NO separate experiment system**
- All experiment logic was in `main.py`:
  - Manual trial loop: `for trial in range(args.n_trials)`
  - Manual algorithm creation with if/elif chains
  - Manual training loop: `while True: alg.step()`
  - Manual evaluation: `if (iteration+1) % 10 == 0: test_reward = ...`
  - Manual result aggregation: `all_rewards.append(rewards)`
  - Manual plotting: `plt.errorbar(...)`
  - Manual file saving: `pickle.dump(weights, ...)`

**Key characteristics:**
- ~220 lines of experiment logic in `main.py`
- No guarantee of fair comparisons (seeds set once at start)
- Different algorithms might get different policy initializations
- Manual metric collection (only rewards)
- Repetitive code for each algorithm type

**Example from old main.py:**
```python
for trial in range(args.n_trials):
    policy = policies.get_policy(args, env)
    if args.alg == 'ES':
        alg = ESModule(...)
    elif args.alg == 'PPO':
        alg = PPOModule(...)
    # ... more if/elif
    
    while True:
        weights = alg.step()
        if (iteration+1) % 10 == 0:
            test_reward = run_func(...)
            rewards.append(test_reward)
    # Manual aggregation, plotting, saving
```

### What We Changed (`experiments/`)

**What we have now:**
- **New `experiments/` module** with:
  - `runner.py` - `ExperimentRunner` class
  - `registry.py` - Algorithm registry

**`experiments/runner.py` - ExperimentRunner class:**
- Encapsulates all experiment logic
- **Ensures fair comparisons:**
  - Sets seeds per trial: `random.seed(self.seed + trial_num)`
  - Creates fresh policy per trial (same initialization)
  - Same evaluation intervals and conditions
- **Automatic metrics collection:**
  - Integrates with `MetricsTracker`
  - Records all metrics from algorithm steps
- **Automatic result handling:**
  - Aggregates across trials
  - Generates plots
  - Saves weights, metrics, results

**`experiments/registry.py` - Algorithm registry:**
- Maps algorithm names to classes
- `get_algorithm(name)` function
- Easy to add new algorithms

**Changes:**
1. ✅ **Extracted experiment logic** from `main.py` to `ExperimentRunner`
2. ✅ **Fair comparison guarantees** - seeds, policy init, evaluation all controlled
3. ✅ **Automatic metrics** - integrates with `MetricsTracker`
4. ✅ **Cleaner main.py** - now ~150 lines vs ~220 lines
5. ✅ **Algorithm registry** - no more if/elif chains

**Why:**
- Ensures fair comparisons (critical for research)
- Reduces code duplication
- Makes it easy to add new algorithms
- Separates concerns (experiment logic vs algorithm logic)

---

## METRICS

### Original

**What she had:**
- **NO metrics system**
- Only tracked:
  - Episode rewards (manually appended to list)
  - Final reward
  - Time to completion
- **No PPO metrics**: No KL divergence, clip fraction, advantage stats
- **No ES metrics**: No variance, gradient norms, population stats
- **No policy metrics**: No weight drift, parameter statistics
- Metrics were just printed or saved as simple arrays

**Key characteristics:**
- Manual tracking: `rewards.append(test_reward)`
- Only reward curves
- No diagnostic information
- Hard to understand why algorithms fail

**Example from old code:**
```python
rewards = []
while True:
    weights = alg.step()
    if (iteration+1) % 10 == 0:
        test_reward = run_func(...)
        rewards.append(test_reward)  # Only reward!
# No KL div, no clip fraction, no variance, etc.
```

### What We Changed (`metrics/`)

**What we have now:**
- **New `metrics/` module** with:
  - `tracker.py` - `MetricsTracker` class

**`metrics/tracker.py` - MetricsTracker class:**
- Centralized metrics collection
- **PPO metrics** (from `PPOUpdater`):
  - KL divergence
  - Clip fraction
  - Policy loss, value loss
  - Entropy
  - Advantage statistics (mean, std)
  - Ratio statistics (mean, std)
- **ES metrics** (from `ESUpdater`):
  - Reward variance
  - Gradient estimate norms
  - Population statistics (mean, std, min, max rewards)
- **Policy metrics**:
  - Weight L2 norm
  - Weight drift from initialization
  - Parameter statistics (mean, std)
- **Automatic saving**: Saves metrics to pickle file

**Changes:**
1. ✅ **Created metrics system** - `MetricsTracker` class
2. ✅ **Comprehensive tracking** - PPO, ES, and policy metrics
3. ✅ **Automatic collection** - algorithms return metrics dicts
4. ✅ **Policy drift tracking** - tracks weight distance from initialization
5. ✅ **Persistent storage** - saves metrics to disk for analysis

**Why:**
- **Diagnosis**: Can see why algorithms fail (KL div too high? Clip fraction? Variance?)
- **Research**: Need detailed metrics to understand algorithm behavior
- **Comparison**: Can compare not just rewards but internal metrics
- **Debugging**: Policy drift helps identify training instability

---

## Summary Table

| Component | Original | What We Changed |
|-----------|----------|-----------------|
| **Policies** | `policies.py` with `get_policy(args, env)` | `policies/continuous.py` with `get_policy(policy_type, state_dim, action_dim)` - more modular, less coupling |
| **Experiments** | All logic in `main.py` (~220 lines) | `experiments/runner.py` - `ExperimentRunner` class ensures fair comparisons |
| **Metrics** | None - only manual reward tracking | `metrics/tracker.py` - `MetricsTracker` class with comprehensive metrics (PPO, ES, policy) |
| **Algorithm Registry** | None - if/elif chains in main.py | `experiments/registry.py` - clean algorithm lookup |

---

## Key Improvements

### 1. Fair Comparisons
**Before:** Seeds set once, different algorithms might get different initializations  
**After:** `ExperimentRunner` ensures same seeds, same policy init, same evaluation for all algorithms

### 2. Comprehensive Metrics
**Before:** Only episode rewards  
**After:** KL divergence, clip fraction, variance, gradient norms, policy drift, etc.

### 3. Modularity
**Before:** Everything in `main.py`  
**After:** Separated into `experiments/`, `metrics/`, `policies/` modules

### 4. Extensibility
**Before:** Adding new algorithm = modify `main.py` with if/elif  
**After:** Add to registry, `ExperimentRunner` handles the rest

---

## What Stayed the Same

- Policy network architectures (`ESPolicyContinuous`, `PPOPolicyContinuous`)
- Environment wrappers (`run_env_ES`, `run_env_PPO`)
- Core algorithm logic (PPO updates, ES updates)
- Command-line interface (backward compatible)

---

## Migration Impact

**For users:**
- Command-line interface unchanged
- Same results (verified with same seeds)
- More metrics available for analysis

**For developers:**
- Easier to add new algorithms
- Fair comparisons guaranteed
- Rich metrics for debugging
- Cleaner code structure

