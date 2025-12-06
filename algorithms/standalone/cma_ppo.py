import copy
import torch
from typing import Dict, Any, Callable
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
        metrics_tracker=None
    ):
        self.policy = policy
        self.env_func = env_func
        self.metrics_tracker = metrics_tracker
        
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

