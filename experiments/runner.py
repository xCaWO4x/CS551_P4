import os
import time
import pickle
import numpy as np
import torch
import random
import sys
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: create a dummy tqdm class
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def update(self, n=1):
            pass
        def set_postfix(self, *args, **kwargs):
            pass
        def write(self, s):
            print(s, flush=True)
        def close(self):
            pass
from algorithms.base import Algorithm
from metrics.tracker import MetricsTracker
from experiments.registry import get_algorithm

class ExperimentRunner:
    """Runs experiments ensuring fair comparisons."""
    
    def __init__(
        self,
        algorithm_name: str,
        algorithm_config: Dict[str, Any],
        policy,
        env_func,
        seed: int = 1234,
        n_trials: int = 5,
        max_iterations: int = 10000,
        eval_interval: int = 10,
        reward_goal: Optional[float] = None,
        consecutive_goal_max: int = 10,
        save_dir: str = './checkpoints',
        metrics_tracker: Optional[MetricsTracker] = None
    ):
        self.algorithm_name = algorithm_name
        self.algorithm_config = algorithm_config
        self.policy = policy
        self.env_func = env_func
        self.seed = seed
        self.n_trials = n_trials
        self.max_iterations = max_iterations
        self.eval_interval = eval_interval
        self.reward_goal = reward_goal
        self.consecutive_goal_max = consecutive_goal_max
        self.save_dir = save_dir
        
        # Set up metrics tracker
        if metrics_tracker is None:
            metrics_tracker = MetricsTracker(save_dir=save_dir)
        self.metrics_tracker = metrics_tracker
    
    def _save_intermediate_plot(self, iterations, rewards, trial_num, current_iter, final=False):
        """Save intermediate plot during training."""
        if not iterations or not rewards:
            return
        
        try:
            # Create trial-specific directory
            trial_dir = os.path.join(self.save_dir, f'trial_{trial_num+1}_plots')
            os.makedirs(trial_dir, exist_ok=True)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, rewards, 'b-', alpha=0.7, label='Reward')
            if len(rewards) > 1:
                # Moving average
                window = min(10, len(rewards) // 2)
                if window > 1:
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    moving_iter = iterations[window-1:]
                    plt.plot(moving_iter, moving_avg, 'r-', linewidth=2, label=f'Moving avg ({window})')
            
            if self.reward_goal:
                plt.axhline(y=self.reward_goal, color='g', linestyle='--', label='Goal')
            
            plt.xlabel('Iteration')
            plt.ylabel('Reward')
            plt.title(f'{self.algorithm_name.upper()} Trial {trial_num+1} - Iter {current_iter}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            suffix = 'final' if final else f'iter_{current_iter}'
            plot_path = os.path.join(trial_dir, f'rewards_{suffix}.png')
            plt.savefig(plot_path, dpi=100)
            plt.close()
            
            # Also save data
            data_path = os.path.join(trial_dir, f'rewards_{suffix}.txt')
            np.savetxt(data_path, np.column_stack([iterations, rewards]), 
                      header='iteration reward', fmt='%d %.4f')
        except Exception as e:
            # Don't fail if plotting fails
            pass
    
    def run_trial(self, trial_num: int) -> Dict[str, Any]:
        """Run a single trial."""
        # Set seeds for reproducibility
        random.seed(self.seed + trial_num)
        np.random.seed(self.seed + trial_num)
        torch.manual_seed(self.seed + trial_num)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed + trial_num)
        
        # Create algorithm
        AlgorithmClass = get_algorithm(self.algorithm_name)
        algorithm = AlgorithmClass(
            policy=self.policy,
            env_func=self.env_func,
            metrics_tracker=self.metrics_tracker,
            **self.algorithm_config
        )
        
        # Set initial policy weights for drift tracking
        self.metrics_tracker.set_initial_policy_weights(algorithm.get_policy())
        
        # Training loop
        start_time = time.time()
        consecutive_goal_count = 0
        iteration = 0
        eval_rewards = []
        eval_iterations = []
        all_metrics = []
        
        # Create progress bar with reward display
        pbar = tqdm(
            total=self.max_iterations,
            desc=f'Trial {trial_num+1}/{self.n_trials}',
            unit='iter',
            file=sys.stdout,
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
        try:
        while iteration < self.max_iterations:
            # Training step
            metrics = algorithm.step()
            all_metrics.append(metrics)
            self.metrics_tracker.record(metrics, iteration)
            
            # Evaluation
            if (iteration + 1) % self.eval_interval == 0:
                # Evaluate policy
                from envs.wrappers import run_env_PPO
                test_reward = run_env_PPO(
                    policy=algorithm.get_policy(),
                    env_func=self.env_func,
                    stochastic=False,
                    reward_only=True
                )
                eval_rewards.append(test_reward)
                    eval_iterations.append(iteration + 1)
                    
                    # Update progress bar with reward info
                    best_reward = max(eval_rewards) if eval_rewards else test_reward
                    pbar.set_postfix({
                        'reward': f'{test_reward:.2f}',
                        'best': f'{best_reward:.2f}',
                        'avg': f'{np.mean(eval_rewards[-10:]):.2f}' if len(eval_rewards) >= 10 else f'{np.mean(eval_rewards):.2f}'
                    })
                    
                    # Also print to ensure it's logged (tqdm.write() goes above progress bar)
                    pbar.write(f'Trial {trial_num+1}, Iter {iteration+1}: reward = {test_reward:.2f}, best = {best_reward:.2f}')
                    
                    # Save intermediate plot every 50 evaluations
                    if len(eval_rewards) % 50 == 0 and self.save_dir:
                        self._save_intermediate_plot(eval_iterations, eval_rewards, trial_num, iteration)
                
                # Check for early stopping
                if self.reward_goal and test_reward >= self.reward_goal:
                    consecutive_goal_count += 1
                    if consecutive_goal_count >= self.consecutive_goal_max:
                            pbar.write(f'Early stopping: reached goal {self.consecutive_goal_max} times')
                        break
                else:
                    consecutive_goal_count = 0
            
                # Update progress bar each iteration
                pbar.update(1)
            iteration += 1
        finally:
            pbar.close()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Final evaluation
        from envs.wrappers import run_env_PPO
        final_reward = run_env_PPO(
            policy=algorithm.get_policy(),
            env_func=self.env_func,
            stochastic=False,
            reward_only=True
        )
        
        return {
            'eval_rewards': eval_rewards,
            'final_reward': final_reward,
            'elapsed_time': elapsed_time,
            'iterations': iteration,
            'algorithm': algorithm
        }
    
    def run(self) -> Dict[str, Any]:
        """Run all trials and aggregate results."""
        all_trial_results = []
        
        for trial in range(self.n_trials):
            print(f'\n=== Trial {trial + 1}/{self.n_trials} ===', flush=True)
            trial_result = self.run_trial(trial)
            all_trial_results.append(trial_result)
            print(f'Trial {trial + 1} completed: final reward = {trial_result["final_reward"]:.2f}, time = {trial_result["elapsed_time"]:.2f}s', flush=True)
        
        # Aggregate results
        all_eval_rewards = [r['eval_rewards'] for r in all_trial_results]
        all_final_rewards = [r['final_reward'] for r in all_trial_results]
        all_times = [r['elapsed_time'] for r in all_trial_results]
        
        # Pad eval rewards to same length
        max_len = max(len(r) for r in all_eval_rewards)
        padded_rewards = []
        for rewards in all_eval_rewards:
            padded = rewards + [self.reward_goal if self.reward_goal else rewards[-1]] * (max_len - len(rewards))
            padded_rewards.append(padded)
        
        padded_rewards = np.array(padded_rewards)
        rewards_mean = np.mean(padded_rewards, axis=0)
        rewards_std = np.std(padded_rewards, axis=0)
        
        # Save results
        exp_dir = os.path.join(self.save_dir, all_trial_results[0]['algorithm'].model_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save weights
        final_algorithm = all_trial_results[-1]['algorithm']
        weights_path = os.path.join(exp_dir, 'weights.pkl')
        final_algorithm.save_checkpoint(weights_path)
        
        # Save metrics
        self.metrics_tracker.save(os.path.join(exp_dir, 'metrics.pkl'))
        
        # Save rewards
        np.savetxt(os.path.join(exp_dir, 'rewards.txt'), rewards_mean)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            np.arange(max_len) * self.eval_interval,
            rewards_mean,
            yerr=rewards_std,
            label='rewards',
            capsize=3
        )
        if self.reward_goal:
            plt.axhline(y=self.reward_goal, color='r', linestyle='--', label='goal')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.title(f'{self.algorithm_name.upper()} Training Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'rewards_plot.png'))
        plt.close()
        
        # Save summary
        total_mean = np.mean(all_final_rewards)
        time_mean = np.mean(all_times)
        
        summary_path = os.path.join(exp_dir, 'results.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Algorithm: {self.algorithm_name}\n")
            f.write(f"Average final reward: {total_mean:.2f}\n")
            f.write(f"Average time: {time_mean:.2f}s\n")
            f.write(f"Trials: {self.n_trials}\n")
            f.write(f"Results saved at: {exp_dir}\n")
        
        print(f'\n=== Results ===', flush=True)
        print(f'Average final reward: {total_mean:.2f}', flush=True)
        print(f'Average time: {time_mean:.2f}s', flush=True)
        print(f'Results saved at: {exp_dir}', flush=True)
        
        return {
            'rewards_mean': rewards_mean,
            'rewards_std': rewards_std,
            'final_rewards': all_final_rewards,
            'times': all_times,
            'exp_dir': exp_dir
        }

