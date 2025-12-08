import matplotlib
matplotlib.use('Agg')
from functools import partial
import logging
import os
import sys
import gymnasium as gym
import numpy as np
from gymnasium import logger as gym_logger
import random
import torch

# Ensure unbuffered output for SLURM (flush after each print)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# New modular imports
from policies import get_policy
from options import parse_args
from experiments import ExperimentRunner
from metrics import MetricsTracker

def get_env(render_mode=None):
    """Create BipedalWalker-v3 environment."""
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    return env

def get_goal():
    """Return reward goal for BipedalWalker."""
    return 250

def main():
    args = parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    gym_logger.setLevel(logging.CRITICAL)
    
    # Create environment function
    env_func = get_env
    env = get_env()
    reward_goal = get_goal()
    
    # Handle rendering mode
    if args.render:
        from envs.wrappers import run_env_PPO
        policy = get_policy(args.alg, state_dim=24, action_dim=4)
        weights_path = os.path.join(args.directory, 'weights.pkl')
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path)
            if 'policy_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['policy_state_dict'])
            else:
                policy.load_state_dict(checkpoint)
        total_reward = run_env_PPO(
            policy=policy,
            env_func=env_func,
            stochastic=False,
            render=True,
            reward_only=True
        )
        print(f"Total rewards from episode: {total_reward}")
        return
    
    # Create policy
    policy = get_policy(args.alg, state_dim=24, action_dim=4)
    
    # Build algorithm config based on args
    algorithm_config = {}
    
    if args.alg.upper() == 'PPO':
        algorithm_config = {
            'n_updates': args.n_updates,
            'batch_size': args.batch_size,
            'max_steps': args.max_steps,
            'gamma': args.gamma,
            'clip': args.clip,
            'ent_coeff': args.ent_coeff,
            'learning_rate': args.ppo_lr if hasattr(args, 'ppo_lr') else args.lr
        }
    elif args.alg.upper() == 'ES':
        algorithm_config = {
            'population_size': args.population_size,
            'sigma': args.sigma,
            'learning_rate': args.es_lr if hasattr(args, 'es_lr') else args.lr,
            'threadcount': args.population_size
        }
    elif args.alg.upper() == 'ESPPO':
        algorithm_config = {
            'population_size': args.population_size,
            'sigma': args.sigma,
            'n_updates': args.n_updates,
            'batch_size': args.batch_size,
            'max_steps': args.max_steps,
            'gamma': args.gamma,
            'clip': args.clip,
            'ent_coeff': args.ent_coeff,
            'n_seq': args.n_seq,
            'ppo_learning_rate': args.ppo_lr,
            'es_learning_rate': args.es_lr,
            'threadcount': args.population_size
        }
    elif args.alg.upper() == 'MAXPPO':
        algorithm_config = {
            'population_size': args.population_size,
            'sigma': args.sigma,
            'n_updates': args.n_updates,
            'batch_size': args.batch_size,
            'max_steps': args.max_steps,
            'gamma': args.gamma,
            'clip': args.clip,
            'ent_coeff': args.ent_coeff,
            'n_seq': args.n_seq,
            'ppo_learning_rate': args.ppo_lr,
            'threadcount': args.population_size
        }
    elif args.alg.upper() == 'ALTPPO':
        algorithm_config = {
            'population_size': args.population_size,
            'sigma': args.sigma,
            'n_updates': args.n_updates,
            'batch_size': args.batch_size,
            'max_steps': args.max_steps,
            'gamma': args.gamma,
            'clip': args.clip,
            'ent_coeff': args.ent_coeff,
            'n_alt': args.n_alt,
            'es_learning_rate': args.es_lr,
            'ppo_learning_rate': args.ppo_lr,
            'threadcount': args.population_size
        }
    elif args.alg.upper() in ['CMA_PPO', 'CMAPPO']:
        algorithm_config = {
            'n_updates': args.n_updates,
            'batch_size': args.batch_size,
            'max_steps': args.max_steps,
            'gamma': args.gamma,
            'lam': args.lam if hasattr(args, 'lam') else 0.95,
            'lr_mean': args.cma_lr_mean if hasattr(args, 'cma_lr_mean') else 3e-4,
            'lr_var': args.cma_lr_var if hasattr(args, 'cma_lr_var') else 3e-4,
            'lr_value': args.cma_lr_value if hasattr(args, 'cma_lr_value') else 1e-3,
            'history_size': args.history_size if hasattr(args, 'history_size') else 5,
            'kernel_std': args.kernel_std if hasattr(args, 'kernel_std') else 0.1
        }
    else:
        raise ValueError(f"Unknown algorithm: {args.alg}")
    
    # Create metrics tracker
    metrics_tracker = MetricsTracker(save_dir=args.directory)
    
    # Create and run experiment
    runner = ExperimentRunner(
        algorithm_name=args.alg.lower(),
        algorithm_config=algorithm_config,
        policy=policy,
        env_func=env_func,
        seed=args.seed,
        n_trials=args.n_trials,
        max_iterations=args.epoch,
        eval_interval=10,
        reward_goal=reward_goal,
        consecutive_goal_max=10,
        save_dir=args.directory,
        metrics_tracker=metrics_tracker
    )
    
    results = runner.run()
    print(f"\nExperiment complete! Results saved to: {results['exp_dir']}")

if __name__ == '__main__':
    main()
