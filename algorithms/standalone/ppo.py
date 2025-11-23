import copy
import torch
from typing import Dict, Any, Callable
from algorithms.base import Algorithm
from algorithms.core.ppo_core import PPOUpdater
from envs.wrappers import run_env_PPO

class PPO(Algorithm):
    """Standalone Proximal Policy Optimization algorithm."""
    
    def __init__(
        self,
        policy: torch.nn.Module,
        env_func: Callable,
        n_updates: int = 5,
        batch_size: int = 64,
        max_steps: int = 256,
        gamma: float = 0.99,
        clip: float = 0.01,
        ent_coeff: float = 0.0,
        learning_rate: float = 0.0001,
        metrics_tracker=None
    ):
        self.policy = policy
        self.env_func = env_func
        self.metrics_tracker = metrics_tracker
        
        # Create PPO updater
        self.ppo_updater = PPOUpdater(
            policy=policy,
            env_func=lambda **kwargs: run_env_PPO(policy=policy, env_func=env_func, **kwargs),
            n_updates=n_updates,
            batch_size=batch_size,
            max_steps=max_steps,
            gamma=gamma,
            clip=clip,
            ent_coeff=ent_coeff,
            learning_rate=learning_rate
        )
    
    def step(self) -> Dict[str, Any]:
        """Perform one PPO training step."""
        # Run PPO updates
        metrics = self.ppo_updater.update(n_sequences=1)
        
        # Evaluate policy
        reward = run_env_PPO(
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
            'optimizer_state_dict': self.ppo_updater.optimizer.state_dict()
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.ppo_updater.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    @property
    def model_name(self) -> str:
        return "PPO_{}_{}_{}_{}_{}_{}_{}".format(
            self.ppo_updater.n_updates,
            self.ppo_updater.batch_size,
            self.ppo_updater.max_steps,
            self.ppo_updater.gamma,
            self.ppo_updater.clip,
            self.ppo_updater.ent_coeff,
            self.ppo_updater.learning_rate
        )

