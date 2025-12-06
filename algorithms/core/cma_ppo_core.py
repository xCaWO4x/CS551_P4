import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from typing import Dict, Any, Callable, List, Tuple
from utils import to_var
from collections import deque

class CMA_PPOUpdater:
    """
    CMA-PPO update logic following Hämäläinen et al. (2018).
    
    Key features:
    - Separate mean and variance networks with separate optimizers
    - History buffer of last H iterations
    - Mirroring of negative-advantage actions around current mean
    - Variance network trained first on history data (rank-µ update)
    - Mean network trained on current iteration data only
    - Unclipped log-likelihood objective weighted by positive/mirrored advantages
    """
    
    def __init__(
        self,
        policy: torch.nn.Module,  # CMAPPOPolicyContinuous
        env_func: Callable,
        n_updates: int = 5,
        batch_size: int = 2048,
        max_steps: int = 16000,  # Total steps per iteration
        gamma: float = 0.99,
        lam: float = 0.95,  # GAE lambda
        lr_mean: float = 3e-4,
        lr_var: float = 3e-4,
        lr_value: float = 1e-3,
        history_size: int = 5,  # H iterations
        kernel_std: float = 0.1  # Gaussian kernel std for mirroring
    ):
        self.policy = policy
        self.env_func = env_func
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.gamma = gamma
        self.lam = lam
        self.lr_mean = lr_mean
        self.lr_var = lr_var
        self.lr_value = lr_value
        self.history_size = history_size
        self.kernel_std = kernel_std
        
        # Separate optimizers for mean, variance, and value networks
        self.optimizer_mean = optim.Adam(self.policy.mean_net.parameters(), lr=self.lr_mean)
        self.optimizer_var = optim.Adam(self.policy.var_net.parameters(), lr=self.lr_var)
        self.optimizer_value = optim.Adam(self.policy.vf.parameters(), lr=self.lr_value)
        
        # History buffer: stores (states, actions_pre_tanh, advantages, returns) for last H iterations
        self.history_buffer = deque(maxlen=history_size)
    
    def compute_gae(self, rewards, values, masks, last_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            masks: Array of done masks (1 if not done, 0 if done)
            last_value: Value estimate for last state
        
        Returns:
            advantages, returns
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        next_value = last_value
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            advantages[step] = gae
            returns[step] = gae + values[step]
            next_value = values[step]
        
        return advantages, returns
    
    def mirror_actions(
        self,
        states: np.ndarray,
        actions_pre_tanh: np.ndarray,
        advantages: np.ndarray,
        means: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mirror negative-advantage actions around current mean with Gaussian kernel weights.
        
        For actions with negative advantages:
        - Mirror: a_mirrored = 2 * μ(s) - a
        - Weight: w = exp(-||a - μ(s)||² / (2 * kernel_std²))
        
        Args:
            states: State array
            actions_pre_tanh: Pre-tanh action array
            advantages: Advantage array
            means: Current mean predictions for states
        
        Returns:
            mirrored_states, mirrored_actions, mirrored_weights
        """
        # Find negative advantage indices
        neg_indices = advantages < 0
        
        if not np.any(neg_indices):
            # No negative advantages, return empty arrays
            return np.array([]), np.array([]), np.array([])
        
        neg_states = states[neg_indices]
        neg_actions = actions_pre_tanh[neg_indices]
        neg_means = means[neg_indices]
        neg_advantages = advantages[neg_indices]
        
        # Ensure shapes are correct - actions and means should be 2D (n_samples, action_dim)
        if neg_actions.ndim == 1:
            neg_actions = neg_actions.reshape(-1, 1)
        elif neg_actions.ndim > 2:
            neg_actions = neg_actions.reshape(neg_actions.shape[0], -1)
        
        if neg_means.ndim == 1:
            neg_means = neg_means.reshape(-1, 1)
        elif neg_means.ndim > 2:
            neg_means = neg_means.reshape(neg_means.shape[0], -1)
        
        # Ensure advantages are 1D
        if neg_advantages.ndim > 1:
            neg_advantages = neg_advantages.squeeze()
        if neg_advantages.ndim == 0:
            neg_advantages = np.array([neg_advantages])
        
        # Mirror actions: a_mirrored = 2 * μ - a
        mirrored_actions = 2 * neg_means - neg_actions
        
        # Compute Gaussian kernel weights: w = exp(-||a - μ||² / (2 * σ²))
        # Sum over action dimensions to get per-sample distance
        action_diff = neg_actions - neg_means
        squared_dist = np.sum(action_diff ** 2, axis=-1)  # Sum over last dimension, shape: (n_neg,)
        kernel_weights = np.exp(-squared_dist / (2 * self.kernel_std ** 2))  # Shape: (n_neg,)
        
        # Use absolute value of negative advantages as additional weighting
        # (we want to weight by how "bad" the action was)
        advantage_weights = np.abs(neg_advantages)  # Shape: (n_neg,)
        
        # Combined weight (both are 1D, so this should work)
        combined_weights = kernel_weights * advantage_weights  # Shape: (n_neg,)
        
        return neg_states, mirrored_actions, combined_weights
    
    def build_cma_batch(
        self,
        states: np.ndarray,
        actions_pre_tanh: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        current_means: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build CMA batch with positive advantages and mirrored negative advantages.
        
        Returns:
            batch_states, batch_actions, batch_advantages, batch_returns
        """
        # Get positive advantage samples
        pos_indices = advantages > 0
        pos_states = states[pos_indices]
        pos_actions = actions_pre_tanh[pos_indices]
        pos_advantages = advantages[pos_indices]
        pos_returns = returns[pos_indices]
        
        # Get mirrored negative advantage samples
        neg_states, mirrored_actions, mirror_weights = self.mirror_actions(
            states, actions_pre_tanh, advantages, current_means
        )
        
        if len(neg_states) > 0 and len(pos_states) > 0:
            # Sample mirrored actions according to weights
            # Normalize weights to probabilities
            weight_sum = mirror_weights.sum()
            if weight_sum > 1e-8:
                probs = mirror_weights / weight_sum
            else:
                # If all weights are zero, use uniform distribution
                probs = np.ones(len(neg_states)) / len(neg_states)
            
            n_samples = min(len(neg_states), len(pos_states))  # Balance positive and mirrored
            if n_samples > 0:
                # Ensure we don't try to sample more than available
                n_samples = min(n_samples, len(neg_states))
                # Count non-zero probabilities
                non_zero_count = np.count_nonzero(probs)
                if non_zero_count < n_samples:
                    # If fewer non-zero entries than requested, sample with replacement or reduce size
                    n_samples = min(n_samples, non_zero_count) if non_zero_count > 0 else len(neg_states)
                
                sample_indices = np.random.choice(
                    len(neg_states), size=n_samples, p=probs, replace=(len(neg_states) < n_samples or non_zero_count < n_samples)
                )
                
                sampled_neg_states = neg_states[sample_indices]
                sampled_mirrored_actions = mirrored_actions[sample_indices]
                # Use positive advantages for mirrored actions (they're now "good")
                neg_adv_indices = np.where(advantages < 0)[0]
                sampled_mirrored_advantages = np.abs(advantages[neg_adv_indices][sample_indices])
                sampled_mirrored_returns = returns[neg_adv_indices][sample_indices]
                
                # Ensure consistent shapes for concatenation
                # Flatten any extra dimensions
                if pos_actions.ndim > 2:
                    pos_actions = pos_actions.reshape(pos_actions.shape[0], -1)
                if sampled_mirrored_actions.ndim > 2:
                    sampled_mirrored_actions = sampled_mirrored_actions.reshape(sampled_mirrored_actions.shape[0], -1)
                if pos_actions.ndim == 1:
                    pos_actions = pos_actions.reshape(-1, 1)
                if sampled_mirrored_actions.ndim == 1:
                    sampled_mirrored_actions = sampled_mirrored_actions.reshape(-1, 1)
                
                # Ensure advantages and returns are 1D
                if pos_advantages.ndim > 1:
                    pos_advantages = pos_advantages.squeeze()
                if sampled_mirrored_advantages.ndim > 1:
                    sampled_mirrored_advantages = sampled_mirrored_advantages.squeeze()
                if pos_returns.ndim > 1:
                    pos_returns = pos_returns.squeeze()
                if sampled_mirrored_returns.ndim > 1:
                    sampled_mirrored_returns = sampled_mirrored_returns.squeeze()
                
                # Combine positive and mirrored samples
                batch_states = np.concatenate([pos_states, sampled_neg_states], axis=0)
                batch_actions = np.concatenate([pos_actions, sampled_mirrored_actions], axis=0)
                batch_advantages = np.concatenate([pos_advantages, sampled_mirrored_advantages], axis=0)
                batch_returns = np.concatenate([pos_returns, sampled_mirrored_returns], axis=0)
            else:
                # Fallback to only positive
                batch_states = pos_states
                batch_actions = pos_actions
                batch_advantages = pos_advantages
                batch_returns = pos_returns
        else:
            # Only positive advantages (or no data)
            if len(pos_states) > 0:
                batch_states = pos_states
                batch_actions = pos_actions
                batch_advantages = pos_advantages
                batch_returns = pos_returns
            else:
                # No positive advantages - return empty arrays
                batch_states = np.array([])
                batch_actions = np.array([])
                batch_advantages = np.array([])
                batch_returns = np.array([])
        
        return batch_states, batch_actions, batch_advantages, batch_returns
    
    def update(self, n_sequences: int = 1) -> Dict[str, Any]:
        """
        Perform CMA-PPO updates.
        
        Returns:
            Dictionary with metrics
        """
        all_metrics = {
            'mean_loss': [],
            'var_loss': [],
            'value_loss': [],
            'advantage_mean': [],
            'advantage_std': [],
        }
        
        for _ in range(n_sequences):
            # Collect trajectories
            states, actions_tanh, actions_pre_tanh, rewards, values, logprobs, returns = self.env_func(
                max_steps=self.max_steps,
                gamma=self.gamma
            )
            
            # Compute GAE advantages
            # Get last value estimate for GAE computation
            if len(states) > 0:
                with torch.no_grad():
                    last_state = to_var(states[-1:])
                    last_value, _, _, _ = self.policy.forward(last_state, stochastic=False, return_pre_tanh=True)
                    last_value = last_value.cpu().item()
            else:
                last_value = 0.0
            
            # Create masks (all 1s for now, assuming no early termination in max_steps collection)
            # In practice, we'd track done flags, but for simplicity assume all steps are valid
            masks = np.ones_like(rewards)
            
            # Compute GAE advantages (ignore the returns from env_func, compute GAE instead)
            advantages, returns_gae = self.compute_gae(rewards, values, masks, last_value)
            
            # Normalize advantages
            advantages_mean = advantages.mean()
            advantages_std = advantages.std()
            if advantages_std > 0:
                advantages = (advantages - advantages_mean) / advantages_std
            else:
                advantages = advantages - advantages_mean
            
            # Get current mean predictions for mirroring
            with torch.no_grad():
                states_tensor = to_var(states)
                current_means, _ = self.policy.get_mean_var(states_tensor)
                current_means = current_means.cpu().numpy()
            
            # Add current iteration to history buffer
            self.history_buffer.append({
                'states': states,
                'actions_pre_tanh': actions_pre_tanh,
                'advantages': advantages,
                'returns': returns_gae
            })
            
            # Build CMA batch for current iteration (for mean network)
            cma_states, cma_actions, cma_advantages, cma_returns = self.build_cma_batch(
                states, actions_pre_tanh, advantages, returns_gae, current_means
            )
            
            # Multiple update epochs
            for update in range(self.n_updates):
                # ===== Update Variance Network (using history data) =====
                if len(self.history_buffer) > 0:
                    # Collect all history data
                    hist_states_list = []
                    hist_actions_list = []
                    hist_advantages_list = []
                    
                    for hist_data in self.history_buffer:
                        hist_states = hist_data['states']
                        hist_actions = hist_data['actions_pre_tanh']
                        hist_advantages = hist_data['advantages']
                        
                        # Get current means for history states
                        with torch.no_grad():
                            hist_states_tensor = to_var(hist_states)
                            hist_means, _ = self.policy.get_mean_var(hist_states_tensor)
                            hist_means = hist_means.cpu().numpy()
                        
                        # Build CMA batch for this history entry
                        h_states, h_actions, h_advantages, _ = self.build_cma_batch(
                            hist_states, hist_actions, hist_advantages,
                            hist_data['returns'], hist_means
                        )
                        
                        if len(h_states) > 0:
                            hist_states_list.append(h_states)
                            hist_actions_list.append(h_actions)
                            hist_advantages_list.append(h_advantages)
                    
                    if len(hist_states_list) > 0:
                        # Concatenate all history
                        all_hist_states = np.concatenate(hist_states_list, axis=0)
                        all_hist_actions = np.concatenate(hist_actions_list, axis=0)
                        all_hist_advantages = np.concatenate(hist_advantages_list, axis=0)
                        
                        # Sample minibatches from history
                        n_samples = min(len(all_hist_states), self.batch_size)
                        indices = np.random.choice(len(all_hist_states), size=n_samples, replace=False)
                        
                        sampled_states = to_var(all_hist_states[indices])
                        sampled_actions = to_var(all_hist_actions[indices])
                        sampled_advantages = to_var(all_hist_advantages[indices])
                        
                        # Get mean and variance
                        mean_pred, var_pred = self.policy.get_mean_var(sampled_states)
                        dist = torch.distributions.Normal(mean_pred, var_pred)
                        
                        # Variance loss: -A * log_prob (only positive advantages)
                        log_prob = dist.log_prob(sampled_actions).sum(dim=1)
                        var_loss = -(sampled_advantages.view(-1) * log_prob).mean()
                        
                        # Update variance network
                        self.optimizer_var.zero_grad()
                        var_loss.backward()
                        self.optimizer_var.step()
                        
                        all_metrics['var_loss'].append(var_loss.item())
                
                # ===== Update Mean Network (using current iteration only) =====
                if len(cma_states) > 0 and len(cma_actions) > 0:
                    # Sample minibatch from current CMA batch
                    n_samples = min(len(cma_states), self.batch_size)
                    indices = np.random.choice(len(cma_states), size=n_samples, replace=False)
                    
                    sampled_states = to_var(cma_states[indices])
                    sampled_actions = to_var(cma_actions[indices])
                    sampled_advantages = to_var(cma_advantages[indices])
                    sampled_returns = to_var(cma_returns[indices])
                    
                    # Get mean and variance
                    mean_pred, var_pred = self.policy.get_mean_var(sampled_states)
                    dist = torch.distributions.Normal(mean_pred, var_pred)
                    
                    # Mean loss: -A * log_prob (unclipped, weighted by advantages)
                    log_prob = dist.log_prob(sampled_actions).sum(dim=1)
                    mean_loss = -(sampled_advantages.view(-1) * log_prob).mean()
                    
                    # Update mean network
                    self.optimizer_mean.zero_grad()
                    mean_loss.backward()
                    self.optimizer_mean.step()
                    
                    all_metrics['mean_loss'].append(mean_loss.item())
                
                # ===== Update Value Network (using current iteration) =====
                # Sample minibatch from current iteration
                n_samples = min(len(states), self.batch_size)
                indices = np.random.choice(len(states), size=n_samples, replace=False)
                
                sampled_states = to_var(states[indices])
                sampled_returns = to_var(returns_gae[indices])
                
                # Value loss
                value_pred = self.policy.vf(sampled_states).squeeze()
                value_loss = F.mse_loss(value_pred, sampled_returns)
                
                # Update value network
                self.optimizer_value.zero_grad()
                value_loss.backward()
                self.optimizer_value.step()
                
                all_metrics['value_loss'].append(value_loss.item())
            
            # Store advantage stats
            all_metrics['advantage_mean'].append(advantages_mean)
            all_metrics['advantage_std'].append(advantages_std)
        
        # Return averaged metrics
        return {
            'cma_ppo/mean_loss': np.mean(all_metrics['mean_loss']) if all_metrics['mean_loss'] else 0.0,
            'cma_ppo/var_loss': np.mean(all_metrics['var_loss']) if all_metrics['var_loss'] else 0.0,
            'cma_ppo/value_loss': np.mean(all_metrics['value_loss']) if all_metrics['value_loss'] else 0.0,
            'cma_ppo/advantage_mean': np.mean(all_metrics['advantage_mean']) if all_metrics['advantage_mean'] else 0.0,
            'cma_ppo/advantage_std': np.mean(all_metrics['advantage_std']) if all_metrics['advantage_std'] else 0.0,
        }

