# CS551_P4: Modular ES+PPO Hybrid Algorithms Framework

Modular reinforcement learning framework for experimenting with Evolution Strategies (ES) and Proximal Policy Optimization (PPO) hybrid algorithms.

## Quick Start

```bash
# Run PPO baseline
python main.py -a PPO --n_updates 5 --batch_size 64 --ppo_lr 0.0001 --n_trials 3

# Run ES-PPO hybrid
python main.py -a ESPPO --population_size 10 --sigma 0.1 --n_seq 1 --n_trials 3

# Run CMA-PPO
python main.py -a CMA_PPO --n_updates 5 --batch_size 2048 --max_steps 16000 --n_trials 3
```

## Documentation

This project has three main documentation files:

1. **[CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md)** - Complete codebase structure, architecture, and design patterns
   - Project structure and organization
   - Algorithm composition patterns
   - Metrics system
   - How to add new algorithms
   - Refactoring history

2. **[CMA_PPO.md](CMA_PPO.md)** - Detailed CMA-PPO implementation documentation
   - Algorithm description and differences from vanilla PPO
   - Implementation details
   - Architecture and update flow
   - Hyperparameters and usage
   - Testing status

3. **[RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md)** - How to run experiments
   - Environment setup
   - SLURM cluster usage
   - Running single experiments
   - Baseline configurations
   - Hyperparameter search
   - Monitoring and troubleshooting

## Algorithms

- **PPO** - Proximal Policy Optimization
- **ES** - Evolution Strategies
- **ESPPO** - ES-PPO hybrid (PPO on each ES candidate)
- **MaxPPO** - Best-of-population selection
- **AltPPO** - Alternating ES/PPO steps
- **CMA-PPO** - Covariance Matrix Adaptation PPO (see `CMA_PPO.md`)

## Key Features

✅ **No Code Duplication** - Shared PPO/ES cores written once, used everywhere  
✅ **Fair Comparisons** - Same seeds, policy init, evaluation guaranteed  
✅ **Rich Metrics** - KL divergence, clip fraction, variance, gradient norms, policy drift  
✅ **Easy Extension** - Add new hybrids in ~50-100 lines  
✅ **Real-time Progress** - tqdm progress bars with reward tracking  
✅ **Periodic Plotting** - Automatic plot generation during training

## Requirements

- Python 3.7+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- tqdm (optional, for progress bars)

## Installation

```bash
# Create conda environment
conda create -n myenv python=3.11 -y
conda activate myenv

# Install dependencies
pip install torch numpy matplotlib gymnasium
pip install gymnasium[box2d]  # Required for BipedalWalker
pip install tqdm  # Optional, for progress bars
```

See [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md) for detailed setup instructions.

## Project Structure

```
CS551_P4/
├── algorithms/          # Algorithm implementations
│   ├── base.py         # Algorithm interface
│   ├── core/           # Shared components (PPO, ES, CMA-PPO cores)
│   ├── standalone/     # Standalone algorithms
│   └── hybrid/         # Hybrid algorithms
├── experiments/        # Experiment management
│   ├── runner.py       # ExperimentRunner class
│   └── registry.py     # Algorithm registry
├── metrics/            # Metrics collection
│   └── tracker.py      # MetricsTracker class
├── policies/           # Policy networks
├── envs/               # Environment utilities
├── utils/              # Utility functions
├── main.py             # Entry point
├── options.py          # Command-line arguments
├── runner.sh           # SLURM job template
├── search.sh           # Hyperparameter search script
└── run_baseline.sh     # Baseline experiment script
```

## Getting Started

1. **Read the documentation**: Start with [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) to understand the codebase structure
2. **Set up environment**: Follow [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md) for environment setup
3. **Run a test**: Try a quick test experiment to verify everything works
4. **Run baselines**: Use `run_baseline.sh` to establish baseline performance

## Contributing

When adding new algorithms:
1. Implement the `Algorithm` interface from `algorithms/base.py`
2. Use existing core components (`PPOUpdater`, `ESUpdater`) when possible
3. Register your algorithm in `experiments/registry.py`
4. See [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) for detailed examples

## License

This project is part of CS551 coursework.
