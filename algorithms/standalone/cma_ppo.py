import copy
import torch
import numpy as np
from typing import Dict, Any, Callable, Optional
from algorithms.base import Algorithm
from algorithms.core.cma_ppo_core import CMA_PPOUpdater
from envs.wrappers import run_env_CMA_PPO

class CMA_PPO(Algorithm):
    """CMA-PPO: Covariance Matrix Adaptation PPO following Hämäläinen et al. (2018)."""
    
    def __init__(
        self,
        policy: torch.nn.Module,
        env_func: Callable,
        n_updates: int = 5,
        batch_size: int = 2048,
        max_steps: int = 16000,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr_mean: float = 3e-4,
        lr_var: float = 3e-4,
        lr_value: float = 1e-3,
        history_size: int = 5,
        kernel_std: float = 0.1,
        metrics_tracker=None,
        # Adaptive history parameters
        history_len_min: int = 1,
        reward_goal: Optional[float] = None,
        reward_high: Optional[float] = None
    ):
        self.policy = policy
        self.env_func = env_func
        self.metrics_tracker = metrics_tracker
        
        # Adaptive history configuration
        self.H_max = history_size
        self.H_min = history_len_min
        self.H = self.H_max
        
        # Use reward_goal if passed explicitly, otherwise None
        self.reward_goal = reward_goal
        # Higher confidence band; default: goal + 50 if not provided
        self.reward_high = reward_high if reward_high is not None else (
            self.reward_goal + 50.0 if self.reward_goal is not None else None
        )
        
        # Track best evaluation reward for adaptive history
        self.best_eval_reward = -float('inf')
        
        # Create CMA-PPO updater
        self.cma_ppo_updater = CMA_PPOUpdater(
            policy=policy,
            env_func=lambda **kwargs: run_env_CMA_PPO(policy=policy, env_func=env_func, **kwargs),
            n_updates=n_updates,
            batch_size=batch_size,
            max_steps=max_steps,
            gamma=gamma,
            lam=lam,
            lr_mean=lr_mean,
            lr_var=lr_var,
            lr_value=lr_value,
            history_size=history_size,
            kernel_std=kernel_std
        )
    
    def step(self) -> Dict[str, Any]:
        """Perform one CMA-PPO training step."""
        # Run CMA-PPO updates
        metrics = self.cma_ppo_updater.update(n_sequences=1)
        
        # Evaluate policy
        reward = run_env_CMA_PPO(
            policy=self.policy,
            env_func=self.env_func,
            stochastic=False,
            reward_only=True
        )
        
        metrics['episode_reward'] = reward
        
        # Add policy metrics if tracker available
        if self.metrics_tracker:
            policy_metrics = self.metrics_tracker.get_policy_metrics(self.policy)
            metrics.update(policy_metrics)
        
        return metrics
    
    def get_policy(self) -> torch.nn.Module:
        return self.policy
    
    def save_checkpoint(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_mean_state_dict': self.cma_ppo_updater.optimizer_mean.state_dict(),
            'optimizer_var_state_dict': self.cma_ppo_updater.optimizer_var.state_dict(),
            'optimizer_value_state_dict': self.cma_ppo_updater.optimizer_value.state_dict()
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'optimizer_mean_state_dict' in checkpoint:
            self.cma_ppo_updater.optimizer_mean.load_state_dict(checkpoint['optimizer_mean_state_dict'])
        if 'optimizer_var_state_dict' in checkpoint:
            self.cma_ppo_updater.optimizer_var.load_state_dict(checkpoint['optimizer_var_state_dict'])
        if 'optimizer_value_state_dict' in checkpoint:
            self.cma_ppo_updater.optimizer_value.load_state_dict(checkpoint['optimizer_value_state_dict'])
    
    def update_history_config(
        self,
        eval_reward: float,
        best_eval_reward: Optional[float] = None
    ):
        """
        Called by ExperimentRunner after each evaluation.
        Adjusts history length H based on performance.
        
        Strategy:
        - Pre-goal (R_best <= R_goal): H = H_max (longer history, more exploration)
        - Post-goal (R_best >= R_high): H = H_min (shorter history, stabilize)
        - In-between: Linear interpolation between H_max and H_min
        
        Args:
            eval_reward: Current evaluation reward
            best_eval_reward: Best evaluation reward seen so far (optional)
        """
        # Guard against None values
        if self.reward_goal is None or self.reward_high is None:
            return
        
        # Guard against invalid configuration (division by zero in interpolation)
        if self.reward_high <= self.reward_goal:
            return
        
        # Ensure cma_ppo_updater exists
        if not hasattr(self, 'cma_ppo_updater') or self.cma_ppo_updater is None:
            return
        
        # Update internal best tracker with explicit None checks
        if best_eval_reward is not None:
            self.best_eval_reward = max(self.best_eval_reward, best_eval_reward)
        elif eval_reward is not None:
            self.best_eval_reward = max(self.best_eval_reward, eval_reward)
        else:
            return  # Both are None, cannot proceed
        
        # Use self.* directly for clarity and to avoid variable shadowing
        # Compute new H based on self.best_eval_reward
        if self.best_eval_reward <= self.reward_goal:
            new_H = self.H_max
        elif self.best_eval_reward >= self.reward_high:
            new_H = self.H_min
        else:
            # Linear interpolation: R_goal -> H_max, R_high -> H_min
            # Safe from division by zero due to guard above
            alpha = (self.reward_high - self.best_eval_reward) / (self.reward_high - self.reward_goal)  # in (0,1)
            new_H = int(np.ceil(self.H_min + (self.H_max - self.H_min) * alpha))
        
        new_H = max(self.H_min, min(self.H_max, new_H))
        
        if new_H != self.H:
            # Resize history buffer in updater
            self.H = new_H
            self.cma_ppo_updater.set_history_size(new_H)
            
            # Optional: print debug
            print(
                f"[CMA-PPO] Adjusted history length H to {self.H} "
                f"(R_best={self.best_eval_reward:.2f}, goal={self.reward_goal:.2f}, high={self.reward_high:.2f})",
                flush=True
            )
    
    @property
    def model_name(self) -> str:
        return "CMA_PPO_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.cma_ppo_updater.n_updates,
            self.cma_ppo_updater.batch_size,
            self.cma_ppo_updater.max_steps,
            self.cma_ppo_updater.gamma,
            self.cma_ppo_updater.lam,
            self.cma_ppo_updater.lr_mean,
            self.cma_ppo_updater.lr_var,
            self.cma_ppo_updater.lr_value,
            self.cma_ppo_updater.history_size,
            self.cma_ppo_updater.kernel_std
        )

