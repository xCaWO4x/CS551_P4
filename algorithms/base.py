from abc import ABC, abstractmethod
import torch
from typing import Dict, Any

class Algorithm(ABC):
    """Base interface for all reinforcement learning algorithms."""
    
    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """
        Perform one training step.
        
        Returns:
            Dictionary of metrics from this step.
        """
        pass
    
    @abstractmethod
    def get_policy(self) -> torch.nn.Module:
        """Return the current policy network."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save algorithm state to file."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        """Load algorithm state from file."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a unique identifier for this algorithm configuration."""
        pass

