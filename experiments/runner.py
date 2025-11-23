import os
import time
import pickle
import numpy as np
import torch
import random
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
        all_metrics = []
        
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
                print(f'Trial {trial_num}, Iter {iteration+1}: reward = {test_reward:.2f}')
                
                # Check for early stopping
                if self.reward_goal and test_reward >= self.reward_goal:
                    consecutive_goal_count += 1
                    if consecutive_goal_count >= self.consecutive_goal_max:
                        print(f'Early stopping: reached goal {self.consecutive_goal_max} times')
                        break
                else:
                    consecutive_goal_count = 0
            
            iteration += 1
        
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
            print(f'\n=== Trial {trial + 1}/{self.n_trials} ===')
            trial_result = self.run_trial(trial)
            all_trial_results.append(trial_result)
        
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
        
        print(f'\n=== Results ===')
        print(f'Average final reward: {total_mean:.2f}')
        print(f'Average time: {time_mean:.2f}s')
        print(f'Results saved at: {exp_dir}')
        
        return {
            'rewards_mean': rewards_mean,
            'rewards_std': rewards_std,
            'final_rewards': all_final_rewards,
            'times': all_times,
            'exp_dir': exp_dir
        }

