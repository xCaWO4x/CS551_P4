import copy
import numpy as np
import torch
from multiprocessing.pool import ThreadPool
from typing import Dict, Any, Callable, List, Tuple
from utils import to_var

class ESUpdater:
    """
    Reusable ES logic that can be used by standalone ES or hybrid algorithms.
    Returns comprehensive metrics for analysis.
    """
    
    def __init__(
        self,
        policy: torch.nn.Module,
        env_func: Callable,
        population_size: int = 50,
        sigma: float = 0.1,
        learning_rate: float = 0.001,
        threadcount: int = 4
    ):
        self.policy = policy
        self.weights = list(policy.parameters())
        self.env_func = env_func
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.pool = ThreadPool(threadcount)
    
    def generate_population(self) -> List[List[torch.Tensor]]:
        """
        Generate a population of perturbed weight vectors.
        
        Returns:
            List of weight lists, one per population member
        """
        epsilons_population = []
        for _ in range(self.population_size):
            epsilons = []
            for weight in self.weights:
                epsilons.append(np.random.randn(*weight.data.size()))
            epsilons_population.append(epsilons)
        
        # Create perturbed weights
        population = []
        for epsilons in epsilons_population:
            perturbed = self._perturb_weights(copy.deepcopy(self.weights), epsilons)
            population.append(perturbed)
        
        return population
    
    def _perturb_weights(self, weights: List[torch.Tensor], epsilons: List[np.ndarray]) -> List[torch.Tensor]:
        """Perturb weights with epsilon noise."""
        new_weights = []
        for i, weight in enumerate(weights):
            perturb = to_var(self.sigma * epsilons[i])
            new_weights.append(weight.data + perturb)
        return new_weights
    
    def evaluate_population(self, population: List[List[torch.Tensor]]) -> List[float]:
        """
        Evaluate a population of weights.
        
        Args:
            population: List of weight lists to evaluate
        
        Returns:
            List of rewards for each population member
        """
        rewards = self.pool.map(
            self.env_func,
            population
        )
        return rewards
    
    def update(self, rewards: List[float], epsilons_population: List[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Update weights based on population rewards.
        
        Args:
            rewards: List of rewards for each population member
            epsilons_population: Optional pre-computed epsilons (if None, will recompute)
        
        Returns:
            Dictionary with ES metrics
        """
        # Normalize rewards
        rewards_array = np.array(rewards)
        if np.std(rewards_array) != 0:
            normalized_rewards = (rewards_array - np.mean(rewards_array)) / np.std(rewards_array)
        else:
            normalized_rewards = rewards_array
        
        # If epsilons not provided, we can't compute gradient norm
        # This is fine for standalone ES, but hybrid algorithms should provide them
        gradient_norms = []
        
        if epsilons_population is not None:
            # Compute gradient estimate for each weight layer
            for index, weight in enumerate(self.weights):
                epsilons = np.array([eps[index] for eps in epsilons_population])
                # Gradient estimate: sum(epsilon_i * R_i)
                grad_estimate = np.dot(epsilons.T, normalized_rewards).T
                grad_norm = np.linalg.norm(grad_estimate)
                gradient_norms.append(grad_norm)
                
                # Update weights
                grad_tensor = to_var(grad_estimate)
                weight.data = weight.data + (
                    self.learning_rate / (self.population_size * self.sigma)
                ) * grad_tensor
        else:
            # For standalone ES, we need to regenerate epsilons
            # This is less efficient but maintains compatibility
            epsilons_population = []
            for _ in range(self.population_size):
                epsilons = []
                for weight in self.weights:
                    epsilons.append(np.random.randn(*weight.data.size()))
                epsilons_population.append(epsilons)
            
            for index, weight in enumerate(self.weights):
                epsilons = np.array([eps[index] for eps in epsilons_population])
                grad_estimate = np.dot(epsilons.T, normalized_rewards).T
                grad_norm = np.linalg.norm(grad_estimate)
                gradient_norms.append(grad_norm)
                
                grad_tensor = to_var(grad_estimate)
                weight.data = weight.data + (
                    self.learning_rate / (self.population_size * self.sigma)
                ) * grad_tensor
        
        # Compute metrics
        metrics = {
            'es/reward_mean': float(np.mean(rewards_array)),
            'es/reward_std': float(np.std(rewards_array)),
            'es/reward_variance': float(np.var(rewards_array)),
            'es/reward_min': float(np.min(rewards_array)),
            'es/reward_max': float(np.max(rewards_array)),
            'es/gradient_norm_mean': float(np.mean(gradient_norms)) if gradient_norms else 0.0,
            'es/gradient_norm_max': float(np.max(gradient_norms)) if gradient_norms else 0.0,
            'es/sigma': self.sigma,
            'es/population_size': self.population_size,
        }
        
        return metrics
    
    def get_epsilons_for_weights(self, weights: List[torch.Tensor]) -> List[np.ndarray]:
        """
        Compute epsilon values that would transform current weights to given weights.
        Useful for hybrid algorithms that modify weights via PPO.
        
        Args:
            weights: Target weights
        
        Returns:
            List of epsilon arrays
        """
        epsilons = []
        for i, (weight, current_weight) in enumerate(zip(weights, self.weights)):
            diff = (weight - current_weight.data).detach().cpu().numpy()
            epsilons.append(diff / self.sigma)
        return epsilons

