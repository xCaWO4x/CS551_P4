import copy
import numpy as np
import torch
from typing import Dict, Any, Callable
from algorithms.base import Algorithm
from algorithms.core.ppo_core import PPOUpdater
from algorithms.core.es_core import ESUpdater
from envs.wrappers import run_env_PPO

class AltPPO(Algorithm):
    """
    Alternating PPO: Alternates between ES steps and PPO steps.
    Every n_alt iterations, does an ES step. Otherwise, does PPO step.
    """
    
    def __init__(
        self,
        policy: torch.nn.Module,
        env_func: Callable,
        population_size: int = 50,
        sigma: float = 0.1,
        n_updates: int = 5,
        batch_size: int = 64,
        max_steps: int = 256,
        gamma: float = 0.99,
        clip: float = 0.01,
        ent_coeff: float = 0.0,
        n_alt: int = 5,
        es_learning_rate: float = 0.001,
        ppo_learning_rate: float = 0.0001,
        threadcount: int = 4,
        metrics_tracker=None
    ):
        self.policy = policy
        self.env_func = env_func
        self.metrics_tracker = metrics_tracker
        self.n_alt = n_alt
        self.counter = 0
        
        # Create ES updater
        self.es_updater = ESUpdater(
            policy=policy,
            env_func=lambda weights: run_env_PPO(
                policy=None,  # Will use weights directly
                env_func=env_func,
                reward_only=True
            ),
            population_size=population_size,
            sigma=sigma,
            learning_rate=es_learning_rate,
            threadcount=threadcount
        )
        
        # Create PPO updater
        self.ppo_updater = PPOUpdater(
            policy=policy,
            env_func=lambda **kwargs: run_env_PPO(
                policy=policy,
                env_func=env_func,
                max_steps=kwargs.get('max_steps', max_steps),
                gamma=kwargs.get('gamma', gamma)
            ),
            n_updates=n_updates,
            batch_size=batch_size,
            max_steps=max_steps,
            gamma=gamma,
            clip=clip,
            ent_coeff=ent_coeff,
            learning_rate=ppo_learning_rate
        )
    
    def step(self) -> Dict[str, Any]:
        """Perform one training step (ES or PPO depending on counter)."""
        if (self.counter % self.n_alt) == 0:
            # ES step
            # Generate population
            population = self.es_updater.generate_population()
            
            # Evaluate population (run PPO on each, then evaluate)
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(self.es_updater.pool._processes if hasattr(self.es_updater.pool, '_processes') else 4)
            
            def evaluate_weights(weights):
                cloned_policy = copy.deepcopy(self.policy)
                for i, weight in enumerate(cloned_policy.parameters()):
                    try:
                        weight.data.copy_(weights[i])
                    except:
                        weight.data.copy_(weights[i].data)
                return run_env_PPO(
                    policy=cloned_policy,
                    env_func=self.env_func,
                    stochastic=False,
                    reward_only=True
                )
            
            rewards = pool.map(evaluate_weights, population)
            pool.close()
            pool.join()
            
            # Generate epsilons for update
            epsilons_population = []
            for weights in population:
                epsilons = []
                for i, (weight, current_weight) in enumerate(zip(weights, self.es_updater.weights)):
                    diff = (weight - current_weight.data).detach().cpu().numpy()
                    epsilons.append(diff / self.es_updater.sigma)
                epsilons_population.append(epsilons)
            
            # ES update
            es_metrics = self.es_updater.update(rewards, epsilons_population)
            
            # Update main policy
            for param, weight in zip(self.policy.parameters(), self.es_updater.weights):
                param.data.copy_(weight.data)
            
            metrics = {
                'episode_reward': float(np.mean(rewards)),
                **es_metrics
            }
        else:
            # PPO step
            ppo_metrics = self.ppo_updater.update(n_sequences=1)
            
            # Evaluate
            reward = run_env_PPO(
                policy=self.policy,
                env_func=self.env_func,
                stochastic=False,
                reward_only=True
            )
            
            metrics = {
                'episode_reward': reward,
                **ppo_metrics
            }
        
        self.counter += 1
        
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
            'sigma': self.es_updater.sigma,
            'counter': self.counter,
            'optimizer_state_dict': self.ppo_updater.optimizer.state_dict()
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'sigma' in checkpoint:
            self.es_updater.sigma = checkpoint['sigma']
        if 'counter' in checkpoint:
            self.counter = checkpoint['counter']
        if 'optimizer_state_dict' in checkpoint:
            self.ppo_updater.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    @property
    def model_name(self) -> str:
        return "AltPPO_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.es_updater.population_size,
            self.es_updater.sigma,
            self.ppo_updater.n_updates,
            self.ppo_updater.batch_size,
            self.ppo_updater.max_steps,
            self.ppo_updater.gamma,
            self.ppo_updater.clip,
            self.ppo_updater.ent_coeff,
            self.es_updater.learning_rate,
            self.ppo_updater.learning_rate,
            self.n_alt
        )

