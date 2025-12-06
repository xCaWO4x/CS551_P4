import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from typing import Dict, Any, Callable
from utils import to_var

class PPOUpdater:
    """
    Reusable PPO update logic that can be used by standalone PPO or hybrid algorithms.
    Returns comprehensive metrics for analysis.
    """
    
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
        learning_rate: float = 0.0001
    ):
        self.policy = policy
        self.env_func = env_func
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.gamma = gamma
        self.clip = clip
        self.ent_coeff = ent_coeff
        self.learning_rate = learning_rate
        
        # Create optimizer for this policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
    
    def update(self, n_sequences: int = 1) -> Dict[str, Any]:
        """
        Perform PPO updates and return metrics.
        
        Args:
            n_sequences: Number of times to collect new data and update (for hybrid algorithms)
        
        Returns:
            Dictionary with PPO metrics
        """
        all_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'advantage_mean': [],
            'advantage_std': [],
            'ratio_mean': [],
            'ratio_std': []
        }
        
        for _ in range(n_sequences):
            # Collect trajectories
            states, actions, rewards, values, logprobs, returns = self.env_func(
                max_steps=self.max_steps,
                gamma=self.gamma
            )
            
            # Compute advantages
            advantages = returns - values
            advantages_mean = advantages.mean()
            advantages_std = advantages.std()
            
            # Normalize advantages
            if advantages_std > 0:
                advantages = (advantages - advantages_mean) / advantages_std
            else:
                advantages = advantages - advantages_mean
            
            # Store old logprobs for KL computation
            old_logprobs = logprobs.copy()
            
            # Multiple update epochs
            for update in range(self.n_updates):
                sampler = BatchSampler(
                    SubsetRandomSampler(list(range(advantages.shape[0]))),
                    batch_size=self.batch_size,
                    drop_last=False
                )
                
                for index in sampler:
                    sampled_states = to_var(states[index])
                    sampled_actions = to_var(actions[index])
                    sampled_logprobs = to_var(old_logprobs[index])
                    sampled_returns = to_var(returns[index])
                    sampled_advs = to_var(advantages[index])
                    
                    # Evaluate current policy
                    new_values, new_logprobs, dist_entropy = self.policy.evaluate(
                        sampled_states, sampled_actions
                    )
                    
                    # Compute ratio and clipped surrogate loss
                    ratio = torch.exp(new_logprobs - sampled_logprobs)
                    sampled_advs = sampled_advs.view(-1, 1)
                    
                    surrogate1 = ratio * sampled_advs
                    surrogate2 = torch.clamp(
                        ratio, 1 - self.clip, 1 + self.clip
                    ) * sampled_advs
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()
                    
                    # Value loss
                    sampled_returns = sampled_returns.view(-1, 1)
                    new_values = new_values.view(-1, 1)
                    value_loss = F.mse_loss(new_values, sampled_returns)
                    
                    # Total loss
                    loss = policy_loss + value_loss - self.ent_coeff * dist_entropy.mean()
                    
                    # Update
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Compute metrics for this batch
                    with torch.no_grad():
                        # KL divergence
                        kl = (sampled_logprobs - new_logprobs).mean().item()
                        
                        # Clip fraction
                        clip_frac = (
                            ((ratio < (1 - self.clip)) | (ratio > (1 + self.clip))).float().mean().item()
                        )
                        
                        # Store metrics
                        all_metrics['policy_loss'].append(policy_loss.item())
                        all_metrics['value_loss'].append(value_loss.item())
                        all_metrics['entropy'].append(dist_entropy.mean().item())
                        all_metrics['kl_divergence'].append(kl)
                        all_metrics['clip_fraction'].append(clip_frac)
                        all_metrics['ratio_mean'].append(ratio.mean().item())
                        all_metrics['ratio_std'].append(ratio.std().item())
            
            # Store advantage stats
            all_metrics['advantage_mean'].append(advantages_mean)
            all_metrics['advantage_std'].append(advantages_std)
        
        # Return averaged metrics
        return {
            'ppo/policy_loss': np.mean(all_metrics['policy_loss']),
            'ppo/value_loss': np.mean(all_metrics['value_loss']),
            'ppo/entropy': np.mean(all_metrics['entropy']),
            'ppo/kl_divergence': np.mean(all_metrics['kl_divergence']),
            'ppo/clip_fraction': np.mean(all_metrics['clip_fraction']),
            'ppo/advantage_mean': np.mean(all_metrics['advantage_mean']),
            'ppo/advantage_std': np.mean(all_metrics['advantage_std']),
            'ppo/ratio_mean': np.mean(all_metrics['ratio_mean']),
            'ppo/ratio_std': np.mean(all_metrics['ratio_std']),
        }

