import copy
import numpy as np
import torch
from typing import Dict, Any, Callable
from algorithms.base import Algorithm
from algorithms.core.ppo_core import PPOUpdater
from algorithms.core.es_core import ESUpdater
from envs.wrappers import run_env_PPO, run_env_ES

class ESPPO(Algorithm):
    """
    ES-PPO hybrid: Runs PPO on each ES population member,
    then uses updated weights to compute new epsilon values for ES update.
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
        n_seq: int = 1,
        ppo_learning_rate: float = 0.0001,
        es_learning_rate: float = 0.001,
        threadcount: int = 4,
        metrics_tracker=None
    ):
        self.policy = policy
        self.env_func = env_func
        self.metrics_tracker = metrics_tracker
        self.n_seq = n_seq
        
        # Create ES updater
        self.es_updater = ESUpdater(
            policy=policy,
            env_func=None,  # We'll handle evaluation ourselves
            population_size=population_size,
            sigma=sigma,
            learning_rate=es_learning_rate,
            threadcount=threadcount
        )
        
        # PPO config (will create updaters per population member)
        self.ppo_config = {
            'n_updates': n_updates,
            'batch_size': batch_size,
            'max_steps': max_steps,
            'gamma': gamma,
            'clip': clip,
            'ent_coeff': ent_coeff,
            'learning_rate': ppo_learning_rate
        }
    
    def _ppo_step(self, weights):
        """Run PPO updates on a policy with given weights."""
        init_weights = [w.clone() for w in self.es_updater.weights]
        
        # Create cloned policy with weights
        cloned_policy = copy.deepcopy(self.policy)
        for i, weight in enumerate(cloned_policy.parameters()):
            try:
                weight.data.copy_(weights[i])
            except:
                weight.data.copy_(weights[i].data)
        
        # Create PPO updater for this policy
        ppo_updater = PPOUpdater(
            policy=cloned_policy,
            env_func=lambda **kwargs: run_env_PPO(
                policy=cloned_policy,
                env_func=self.env_func,
                **kwargs
            ),
            **self.ppo_config
        )
        
        # Run PPO updates
        ppo_metrics = ppo_updater.update(n_sequences=self.n_seq)
        
        # Evaluate final policy
        reward = run_env_PPO(
            policy=cloned_policy,
            env_func=self.env_func,
            stochastic=False,
            reward_only=True
        )
        
        # Get final weights and compute new epsilons
        final_weights = list(cloned_policy.parameters())
        new_epsilons = self.es_updater.get_epsilons_for_weights(final_weights)
        
        return reward, new_epsilons, ppo_metrics
    
    def step(self) -> Dict[str, Any]:
        """Perform one ES-PPO training step."""
        # Generate ES population
        population = self.es_updater.generate_population()
        
        # Run PPO on each population member (parallel)
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(self.es_updater.pool._processes if hasattr(self.es_updater.pool, '_processes') else 4)
        results = pool.map(self._ppo_step, population)
        pool.close()
        pool.join()
        
        rewards = [r[0] for r in results]
        new_epsilons_population = [r[1] for r in results]
        ppo_metrics_list = [r[2] for r in results]
        
        # ES update using new epsilons
        es_metrics = self.es_updater.update(rewards, new_epsilons_population)
        
        # Update main policy weights
        for param, weight in zip(self.policy.parameters(), self.es_updater.weights):
            param.data.copy_(weight.data)
        
        # Aggregate PPO metrics
        avg_ppo_metrics = {}
        for key in ppo_metrics_list[0].keys():
            avg_ppo_metrics[key] = np.mean([m[key] for m in ppo_metrics_list])
        
        # Combine metrics
        metrics = {
            'episode_reward': float(np.mean(rewards)),
            **avg_ppo_metrics,
            **es_metrics
        }
        
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
            'es_learning_rate': self.es_updater.learning_rate
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'sigma' in checkpoint:
            self.es_updater.sigma = checkpoint['sigma']
    
    @property
    def model_name(self) -> str:
        return "ESPPO_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.es_updater.population_size,
            self.es_updater.sigma,
            self.ppo_config['n_updates'],
            self.ppo_config['batch_size'],
            self.ppo_config['max_steps'],
            self.ppo_config['gamma'],
            self.ppo_config['clip'],
            self.ppo_config['ent_coeff'],
            self.ppo_config['learning_rate'],
            self.es_updater.learning_rate,
            self.n_seq
        )

