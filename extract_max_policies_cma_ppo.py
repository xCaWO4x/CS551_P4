#!/usr/bin/env python3
"""
Extract policies at maximum reward points for CMA-PPO trials and run greedy evaluations.

This script:
1. Finds the max reward and iteration for each trial (only trials with max > 100)
2. Re-runs training up to that iteration to get the policy state
3. Runs 10 greedy evaluations on each max-reward policy
4. Saves statistics to files
"""

import os
import sys
import numpy as np
import torch
import random
import logging
import gymnasium as gym
from gymnasium import logger as gym_logger

# Ensure unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
sys.stdout.flush()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policies.continuous import get_policy
from algorithms.standalone.cma_ppo import CMA_PPO
from envs.wrappers import run_env_CMA_PPO

def get_env(render_mode=None):
    """Create BipedalWalker-v3 environment."""
    import gymnasium as gym
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    return env

def find_max_reward_iteration(rewards_file):
    """Find the iteration where max reward occurred."""
    rewards = np.loadtxt(rewards_file)
    if rewards.ndim == 0:
        rewards = np.array([rewards])
    max_idx = np.argmax(rewards)
    max_reward = rewards[max_idx]
    # rewards.txt is evaluated every 10 iterations, starting at iteration 10
    max_iteration = (max_idx + 1) * 10
    return max_reward, max_iteration, len(rewards) * 10

def run_training_to_iteration(trial_num, seed, max_iteration, config):
    """Run CMA-PPO training up to max_iteration and return the policy."""
    print(f"\n{'='*80}")
    print(f"Trial {trial_num}: Training up to iteration {max_iteration}")
    print(f"{'='*80}")
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gym_logger.setLevel(logging.CRITICAL)
    
    # Create environment function
    env_func = get_env
    
    # Create policy
    policy = get_policy('CMA_PPO', state_dim=24, action_dim=4)
    
    # Create CMA-PPO algorithm
    algorithm = CMA_PPO(
        policy=policy,
        env_func=env_func,
        n_updates=config['n_updates'],
        batch_size=config['batch_size'],
        max_steps=config['max_steps'],
        gamma=config['gamma'],
        lam=config['lam'],
        lr_mean=config['lr_mean'],
        lr_var=config['lr_var'],
        lr_value=config['lr_value'],
        history_size=config['history_size'],
        kernel_std=config['kernel_std']
    )
    
    # Run training up to max_iteration
    # Note: evaluation happens every 10 iterations, so we need to run (max_iteration // 10) steps
    num_steps = max_iteration // 10
    print(f"Running {num_steps} training steps (evaluated every 10 iterations)...")
    sys.stdout.flush()
    
    for step in range(num_steps):
        metrics = algorithm.step()
        if (step + 1) % 10 == 0 or (step + 1) % 5 == 0:
            reward = metrics.get('episode_reward', 0)
            print(f"  Step {step+1}/{num_steps}: reward = {reward:.2f}")
            sys.stdout.flush()
    
    print(f"Training complete. Policy at iteration {max_iteration} ready.")
    return algorithm.get_policy()

def run_greedy_evaluations(policy, env_func, n_eval=10):
    """Run n_eval greedy evaluations on the policy."""
    test_rewards = []
    print(f"\nRunning {n_eval} greedy evaluations...")
    
    for test_num in range(n_eval):
        reward = run_env_CMA_PPO(
            policy=policy,
            env_func=env_func,
            stochastic=False,
            reward_only=True
        )
        test_rewards.append(reward)
        print(f"  Test {test_num+1}/{n_eval}: reward = {reward:.2f}")
    
    return np.array(test_rewards)

def main():
    base_dir = 'independent_trials_20251207_203413'
    
    # CMA-PPO configuration (from submit script)
    cma_config = {
        'n_updates': 78,
        'batch_size': 1024,
        'max_steps': 4096,
        'gamma': 0.99,
        'lam': 0.95,
        'lr_mean': 3e-4,
        'lr_var': 3e-4,
        'lr_value': 1e-3,
        'history_size': 5,
        'kernel_std': 0.1
    }
    
    # Base seed (trial 1 uses 1235, trial 2 uses 1236, etc.)
    base_seed = 1234
    
    # Find trials with max reward > 100
    trials_to_process = []
    for trial_num in range(1, 6):
        rewards_file = f'{base_dir}/cma_ppo_trial_{trial_num}/CMA_PPO_78_1024_4096_0.99_0.95_0.0003_0.0003_0.001_5_0.1/rewards.txt'
        if os.path.exists(rewards_file):
            max_reward, max_iteration, total_iterations = find_max_reward_iteration(rewards_file)
            if max_reward > 100:
                # Check if already tested (trial 2 hit threshold and was tested)
                stats_file = f'{base_dir}/cma_ppo_trial_{trial_num}/trial_1_max_reward_test_stats.txt'
                if not os.path.exists(stats_file):
                    trials_to_process.append({
                        'trial_num': trial_num,
                        'seed': base_seed + trial_num,
                        'max_reward': max_reward,
                        'max_iteration': max_iteration,
                        'total_iterations': total_iterations,
                        'rewards_file': rewards_file
                    })
                else:
                    print(f"Trial {trial_num}: Already tested (max reward: {max_reward:.2f} at iteration {max_iteration})")
            else:
                print(f"Trial {trial_num}: Max reward {max_reward:.2f} <= 100, skipping")
        else:
            print(f"Trial {trial_num}: rewards.txt not found, skipping")
    
    print(f"\n{'='*80}")
    print(f"Found {len(trials_to_process)} trials to process")
    print(f"{'='*80}")
    
    # Process each trial
    env_func = get_env
    results = []
    
    for trial_info in trials_to_process:
        trial_num = trial_info['trial_num']
        seed = trial_info['seed']
        max_reward = trial_info['max_reward']
        max_iteration = trial_info['max_iteration']
        
        try:
            # Run training up to max iteration
            policy = run_training_to_iteration(trial_num, seed, max_iteration, cma_config)
            
            # Run greedy evaluations
            test_rewards = run_greedy_evaluations(policy, env_func, n_eval=10)
            
            # Calculate statistics
            mean_reward = np.mean(test_rewards)
            std_reward = np.std(test_rewards)
            min_reward = np.min(test_rewards)
            max_reward_eval = np.max(test_rewards)
            median_reward = np.median(test_rewards)
            
            # Print statistics
            print(f"\n{'='*80}")
            print(f"Trial {trial_num} - Max Reward Policy Test Statistics (10 greedy evaluations)")
            print(f"{'='*80}")
            print(f"Max reward during training: {max_reward:.2f} at iteration {max_iteration}")
            print(f"\nTest Statistics:")
            print(f"  Mean:   {mean_reward:.2f}")
            print(f"  Std:    {std_reward:.2f}")
            print(f"  Min:    {min_reward:.2f}")
            print(f"  Max:    {max_reward_eval:.2f}")
            print(f"  Median: {median_reward:.2f}")
            print(f"  All rewards: {[f'{r:.2f}' for r in test_rewards]}")
            print(f"{'='*80}\n")
            
            # Save statistics to file
            trial_dir = f'{base_dir}/cma_ppo_trial_{trial_num}'
            stats_path = os.path.join(trial_dir, f'trial_1_max_reward_test_stats.txt')
            with open(stats_path, 'w') as f:
                f.write(f'Max Reward Policy Test Statistics (10 greedy evaluations)\n')
                f.write(f'Max reward during training: {max_reward:.2f}\n')
                f.write(f'Max reward at iteration: {max_iteration}\n')
                f.write(f'\n')
                f.write(f'Statistics:\n')
                f.write(f'  Mean:   {mean_reward:.2f}\n')
                f.write(f'  Std:    {std_reward:.2f}\n')
                f.write(f'  Min:    {min_reward:.2f}\n')
                f.write(f'  Max:    {max_reward_eval:.2f}\n')
                f.write(f'  Median: {median_reward:.2f}\n')
                f.write(f'  All rewards: {[f"{r:.2f}" for r in test_rewards]}\n')
            
            print(f"Saved statistics to {stats_path}\n")
            
            # Save policy at max point
            policy_path = os.path.join(trial_dir, f'trial_1_max_reward_policy.pkl')
            torch.save({'policy_state_dict': policy.state_dict()}, policy_path)
            print(f"Saved policy to {policy_path}\n")
            
            results.append({
                'trial': trial_num,
                'max_reward': max_reward,
                'max_iteration': max_iteration,
                'test_mean': mean_reward,
                'test_std': std_reward,
                'test_min': min_reward,
                'test_max': max_reward_eval,
                'test_median': median_reward
            })
            
        except Exception as e:
            print(f"\nERROR processing Trial {trial_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Trial':<8} {'Max Reward':<12} {'Max Iter':<10} {'Test Mean':<12} {'Test Std':<12} {'Test Min':<12} {'Test Max':<12}")
        print(f"{'-'*80}")
        for r in results:
            print(f"{r['trial']:<8} {r['max_reward']:<12.2f} {r['max_iteration']:<10} {r['test_mean']:<12.2f} {r['test_std']:<12.2f} {r['test_min']:<12.2f} {r['test_max']:<12.2f}")
        print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

