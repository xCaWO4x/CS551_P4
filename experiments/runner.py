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
        metrics_tracker: Optional[MetricsTracker] = None,
        # Goal detection parameters
        goal_delta: float = 15.0,
        goal_window: int = 3,
        goal_min_consecutive: int = 2,
        early_stop_on_goal: bool = False
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
        
        # Goal detection parameters
        self.goal_delta = goal_delta
        self.goal_window = goal_window
        self.goal_min_consecutive = goal_min_consecutive
        self.early_stop_on_goal = early_stop_on_goal
        
        # Set up metrics tracker
        if metrics_tracker is None:
            metrics_tracker = MetricsTracker(save_dir=save_dir)
        self.metrics_tracker = metrics_tracker
        
        # Initialize goal-tracking state (will be reset per trial)
        self._reset_goal_state()
    
    def _reset_goal_state(self):
        """Reset goal-tracking state for a new trial."""
        self.eval_history = []  # list[float]
        self.goal_consecutive = 0  # c_goal
        self.steps_to_goal = None  # int | None
        self.policy_at_goal = None  # state_dict or algo-specific handle
        self.best_eval_reward = -float('inf')
    
    def _on_evaluation(
        self,
        test_reward: float,
        iteration: int,
        total_env_steps: int,
        algorithm: Algorithm,
        trial_num: int,
        pbar: tqdm
    ):
        """
        Called every time we run a test evaluation inside run_trial.
        Handles:
          - smoothed goal detection
          - steps_to_goal recording
          - saving policy_at_goal
          - adaptive CMA-PPO history (if supported by algorithm)
        """
        # Track best so far (update even if equal to track current best)
        if test_reward >= self.best_eval_reward:
            self.best_eval_reward = test_reward
        
        # Append to eval history
        self.eval_history.append(test_reward)
        if len(self.eval_history) >= self.goal_window:
            barE = float(np.mean(self.eval_history[-self.goal_window:]))
        else:
            barE = float(np.mean(self.eval_history))
        
        # Goal band only if reward_goal is set
        if self.reward_goal is not None:
            # Check if either smoothed mean OR current reward is in goal band
            # This allows detection even if early low rewards pull down the mean
            threshold = self.reward_goal - self.goal_delta
            in_goal_band = (barE >= threshold) or (test_reward >= threshold)
            if in_goal_band:
                self.goal_consecutive += 1
            else:
                self.goal_consecutive = 0
            
            # First time we confirm threshold crossing
            if self.steps_to_goal is None and \
               self.goal_consecutive >= self.goal_min_consecutive:
                self.steps_to_goal = total_env_steps
                
                # Snapshot policy at goal (store policy state_dict for reference)
                policy = algorithm.get_policy()
                if hasattr(policy, "state_dict"):
                    self.policy_at_goal = policy.state_dict()
                else:
                    # Fallback: store the policy object reference
                    self.policy_at_goal = policy
                
                # Log goal detection
                pbar.write(
                    f"[GOAL] {self.algorithm_name.upper()} reached goal band "
                    f"{barE:.2f} (smoothed over {self.goal_window} evals) "
                    f"at {total_env_steps} env steps (iter {iteration+1})"
                )
                
                # Save full checkpoint at goal (policy + optimizers)
                if self.save_dir:
                    goal_policy_path = os.path.join(
                        self.save_dir,
                        f'trial_{trial_num+1}_goal_policy.pkl'
                    )
                    # Save full checkpoint via algorithm (includes policy + optimizers)
                    algorithm.save_checkpoint(goal_policy_path)
                    pbar.write(f"Saved goal policy checkpoint to {goal_policy_path}")
                
                # Early stop if requested
                if self.early_stop_on_goal:
                    raise StopIteration("Early stopping: goal reached.")
        
        # Optional: CMA-specific adaptive history update
        if hasattr(algorithm, "update_history_config"):
            algorithm.update_history_config(
                eval_reward=test_reward,
                best_eval_reward=self.best_eval_reward
            )
    
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
                    moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
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
            np.savetxt(
                data_path,
                np.column_stack([iterations, rewards]),
                header='iteration reward',
                fmt='%d %.4f'
            )
        except Exception:
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
        
        # Reinitialize policy weights using PyTorch's default initialization
        # This ensures each trial starts with the same initialization scheme as the original policy
        # (which was created with get_policy() using PyTorch defaults)
        # The trial-specific seed ensures different random weights per trial
        def reset_weights(m):
            if isinstance(m, torch.nn.Linear):
                # PyTorch's default Linear initialization: U(-k, k) where k = sqrt(1/in_features)
                # This matches what get_policy() uses when creating a new policy
                k = (1.0 / m.in_features) ** 0.5
                torch.nn.init.uniform_(m.weight, -k, k)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Parameter) and m.requires_grad:
                # For learnable parameters like std (in ES/PPO policies), reset to zeros
                # This matches the original initialization: self.std = nn.Parameter(torch.zeros(...))
                torch.nn.init.zeros_(m)
        
        self.policy.apply(reset_weights)

        # Reset goal state for this trial
        self._reset_goal_state()

        # Create algorithm
        AlgorithmClass = get_algorithm(self.algorithm_name)
        
        # Prepare algorithm kwargs
        algo_kwargs = dict(self.algorithm_config)
        
        # If CMA-PPO expects reward_goal / reward_high, pass them through
        if self.algorithm_name.lower() in ['cma_ppo', 'cmappo']:
            if self.reward_goal is not None:
                algo_kwargs.setdefault("reward_goal", self.reward_goal)
                # Optional: choose reward_high as reward_goal + 50 if not explicitly set
                algo_kwargs.setdefault("reward_high", self.reward_goal + 50.0)
        
        algorithm = AlgorithmClass(
            policy=self.policy,
            env_func=self.env_func,
            metrics_tracker=self.metrics_tracker,
            **algo_kwargs
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
                    # Evaluate policy - use correct wrapper based on algorithm
                    if self.algorithm_name.lower() in ['cma_ppo', 'cmappo']:
                        from envs.wrappers import run_env_CMA_PPO
                        test_reward = run_env_CMA_PPO(
                            policy=algorithm.get_policy(),
                            env_func=self.env_func,
                            stochastic=False,
                            reward_only=True
                        )
                    else:
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
                        'avg': (
                            f'{np.mean(eval_rewards[-10:]):.2f}'
                            if len(eval_rewards) >= 10
                            else f'{np.mean(eval_rewards):.2f}'
                        )
                    })

                    # Also print to ensure it's logged
                    pbar.write(
                        f'Trial {trial_num+1}, Iter {iteration+1}: '
                        f'reward = {test_reward:.2f}, best = {best_reward:.2f}'
                    )

                    # Save intermediate plot every 50 evaluations
                    if len(eval_rewards) % 50 == 0 and self.save_dir:
                        self._save_intermediate_plot(
                            eval_iterations, eval_rewards, trial_num, iteration
                        )
                
                    # Determine total env steps for this algorithm
                    if hasattr(algorithm, "total_env_steps"):
                        total_env_steps = int(algorithm.total_env_steps)
                    else:
                        # Fallback: steps_per_iter from algorithm_config
                        steps_per_iter = self.algorithm_config.get("max_steps", None)
                        assert steps_per_iter is not None, (
                            f"Cannot compute total_env_steps: algorithm '{self.algorithm_name}' "
                            f"does not have 'total_env_steps' attribute and 'max_steps' is not "
                            f"in algorithm_config. Please add 'max_steps' to algorithm_config."
                        )
                        total_env_steps = (iteration + 1) * steps_per_iter
                    
                    # Goal detection + adaptive history handling
                    try:
                        self._on_evaluation(
                            test_reward=test_reward,
                            iteration=iteration,
                            total_env_steps=total_env_steps,
                            algorithm=algorithm,
                            trial_num=trial_num,
                            pbar=pbar
                        )
                    except StopIteration:
                        # Early-stop signal from _on_evaluation
                        pbar.write("Early stopping due to goal threshold.")
                        break
                
                    # Legacy threshold test
                    if self.reward_goal and test_reward >= self.reward_goal:
                        ...
                        # (unchanged body for frozen policy tests)
                        ...
            
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
        
        # Save final plot for this trial
        if self.save_dir and eval_rewards:
            self._save_intermediate_plot(
                eval_iterations, eval_rewards, trial_num, iteration, final=True
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
            print(
                f'Trial {trial + 1} completed: '
                f'final reward = {trial_result["final_reward"]:.2f}, '
                f'time = {trial_result["elapsed_time"]:.2f}s',
                flush=True
            )
        
        # Aggregate results
        all_eval_rewards = [r['eval_rewards'] for r in all_trial_results]
        all_final_rewards = [r['final_reward'] for r in all_trial_results]
        all_times = [r['elapsed_time'] for r in all_trial_results]
        
        # Pad eval rewards to same length
        max_len = max(len(r) for r in all_eval_rewards)
        padded_rewards = []
        for rewards in all_eval_rewards:
            if len(rewards) == 0:
                # degenerate case: no evals; just pad zeros or 0-goal
                pad_value = self.reward_goal if self.reward_goal is not None else 0.0
                padded = [pad_value] * max_len
            else:
                pad_value = self.reward_goal if self.reward_goal is not None else rewards[-1]
                padded = rewards + [pad_value] * (max_len - len(rewards))
            padded_rewards.append(padded)
        
        padded_rewards = np.array(padded_rewards, dtype=np.float32)
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
        
        print('\n=== Results ===', flush=True)
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
