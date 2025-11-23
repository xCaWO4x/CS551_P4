import copy
import torch
from typing import Dict, Any, Callable
from algorithms.base import Algorithm
from algorithms.core.es_core import ESUpdater
from envs.wrappers import run_env_ES

class ES(Algorithm):
    """Standalone Evolution Strategies algorithm."""
    
    def __init__(
        self,
        policy: torch.nn.Module,
        env_func: Callable,
        population_size: int = 50,
        sigma: float = 0.1,
        learning_rate: float = 0.001,
        threadcount: int = 4,
        metrics_tracker=None
    ):
        self.policy = policy
        self.env_func = env_func
        self.metrics_tracker = metrics_tracker
        
        # Create ES updater
        self.es_updater = ESUpdater(
            policy=policy,
            env_func=lambda weights: run_env_ES(
                weights=weights,
                policy=policy,
                env_func=env_func,
                render=False,
                stochastic=False
            ),
            population_size=population_size,
            sigma=sigma,
            learning_rate=learning_rate,
            threadcount=threadcount
        )
    
    def step(self) -> Dict[str, Any]:
        """Perform one ES training step."""
        # Generate population
        population = self.es_updater.generate_population()
        
        # Evaluate population
        rewards = self.es_updater.evaluate_population(population)
        
        # Generate epsilons for metrics
        epsilons_population = []
        for weights in population:
            # Compute epsilons that would transform current weights to these weights
            epsilons = []
            for i, (weight, current_weight) in enumerate(zip(weights, self.es_updater.weights)):
                diff = (weight - current_weight.data).cpu().numpy()
                epsilons.append(diff / self.es_updater.sigma)
            epsilons_population.append(epsilons)
        
        # Update weights
        metrics = self.es_updater.update(rewards, epsilons_population)
        
        # Evaluate final policy
        reward = run_env_ES(
            weights=list(self.policy.parameters()),
            policy=self.policy,
            env_func=self.env_func,
            stochastic=False,
            render=False
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
            'sigma': self.es_updater.sigma,
            'learning_rate': self.es_updater.learning_rate
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'sigma' in checkpoint:
            self.es_updater.sigma = checkpoint['sigma']
    
    @property
    def model_name(self) -> str:
        return "ES_{}_{}_{}".format(
            self.es_updater.population_size,
            self.es_updater.sigma,
            self.es_updater.learning_rate
        )

