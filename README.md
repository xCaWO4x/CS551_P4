# CS551_P4: Modular ES+PPO Hybrid Algorithms Framework

Modular reinforcement learning framework for experimenting with Evolution Strategies (ES) and Proximal Policy Optimization (PPO) hybrid algorithms.

## Quick Start

```bash
# Run PPO baseline
python main.py -a PPO --n_updates 5 --batch_size 64 --ppo_lr 0.0001 --n_trials 3

# Run ES-PPO hybrid
python main.py -a ESPPO --population_size 10 --sigma 0.1 --n_seq 1 --n_trials 3

# Run CMA-PPO with adaptive history
python main.py -a CMA_PPO --n_updates 78 --batch_size 1024 --max_steps 4096 --history_size 5 --history_len_min 1 --n_trials 3
```

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

## Algorithms

- **PPO** - Proximal Policy Optimization
- **ES** - Evolution Strategies
- **ESPPO** - ES-PPO hybrid (PPO on each ES candidate)
- **MaxPPO** - Best-of-population selection
- **AltPPO** - Alternating ES/PPO steps
- **CMA-PPO** - Covariance Matrix Adaptation PPO with adaptive history scheduling and goal detection

## Key Features

✅ **No Code Duplication** - Shared PPO/ES cores written once, used everywhere  
✅ **Fair Comparisons** - Same seeds, policy init, evaluation guaranteed  
✅ **Rich Metrics** - KL divergence, clip fraction, variance, gradient norms, policy drift  
✅ **Easy Extension** - Add new hybrids in ~50-100 lines  
✅ **Real-time Progress** - tqdm progress bars with reward tracking  
✅ **Periodic Plotting** - Automatic plot generation during training  
✅ **Adaptive History** - CMA-PPO automatically adjusts history length based on performance  
✅ **Goal Detection** - Automatic policy checkpointing when goal is reached

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
└── plot.py             # Plotting utilities
```

## Documentation

- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** - Complete technical documentation including:
  - Codebase architecture and design patterns
  - CMA-PPO implementation details
  - Adaptive history scheduling
  - Goal detection and stabilization
  - Running experiments on SLURM clusters
  - Algorithm comparison and configuration

## Requirements

- Python 3.7+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- tqdm (optional, for progress bars)

## Getting Started

1. **Set up environment**: Follow the installation instructions above
2. **Read the documentation**: See [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) for detailed information
3. **Run a test**: Try a quick test experiment to verify everything works
4. **Run baselines**: Use `run_baseline.sh` to establish baseline performance

## Contributing

When adding new algorithms:
1. Implement the `Algorithm` interface from `algorithms/base.py`
2. Use existing core components (`PPOUpdater`, `ESUpdater`) when possible
3. Register your algorithm in `experiments/registry.py`
4. See [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) for detailed examples

## License

This project is part of CS551 coursework.
