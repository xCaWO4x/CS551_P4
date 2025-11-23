import torch
import numpy as np
from typing import Dict, Any, List, Optional
import os
import pickle

class MetricsTracker:
    """Centralized metrics collection and tracking."""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir
        self.metrics_history: Dict[str, List[float]] = {}
        self.initial_policy_weights: Optional[List[torch.Tensor]] = None
    
    def record(self, metrics: Dict[str, Any], iteration: int):
        """
        Record metrics for an iteration.
        
        Args:
            metrics: Dictionary of metric name -> value
            iteration: Current iteration number
        """
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            
            # Convert to float if tensor
            if isinstance(value, torch.Tensor):
                value = float(value.item())
            elif isinstance(value, np.ndarray):
                value = float(value.item()) if value.size == 1 else value
            
            self.metrics_history[key].append(value)
    
    def set_initial_policy_weights(self, policy: torch.nn.Module):
        """Store initial policy weights for drift tracking."""
        self.initial_policy_weights = [
            param.data.clone() for param in policy.parameters()
        ]
    
    def get_policy_metrics(self, policy: torch.nn.Module) -> Dict[str, float]:
        """
        Compute policy-specific metrics.
        
        Args:
            policy: Current policy network
        
        Returns:
            Dictionary of policy metrics
        """
        metrics = {}
        
        # Weight statistics
        all_params = []
        for param in policy.parameters():
            all_params.append(param.data.cpu().numpy().flatten())
        
        if all_params:
            all_params = np.concatenate(all_params)
            metrics['policy/param_mean'] = float(np.mean(all_params))
            metrics['policy/param_std'] = float(np.std(all_params))
            metrics['policy/weight_l2_norm'] = float(np.linalg.norm(all_params))
        
        # Weight drift from initialization
        if self.initial_policy_weights is not None:
            drift = 0.0
            for param, init_param in zip(policy.parameters(), self.initial_policy_weights):
                diff = (param.data - init_param).cpu().numpy()
                drift += np.sum(diff ** 2)
            metrics['policy/weight_l2_distance_from_init'] = float(np.sqrt(drift))
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all metrics."""
        summary = {}
        for key, values in self.metrics_history.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1] if values else None
                }
        return summary
    
    def save(self, path: Optional[str] = None):
        """Save metrics to file."""
        if path is None:
            path = os.path.join(self.save_dir, 'metrics.pkl') if self.save_dir else 'metrics.pkl'
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'metrics_history': self.metrics_history,
                'summary': self.get_summary()
            }, f)
    
    def load(self, path: str):
        """Load metrics from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.metrics_history = data['metrics_history']

