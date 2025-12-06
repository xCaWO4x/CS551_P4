import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ESPolicyContinuous(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ESPolicyContinuous, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, action_dim),
            nn.Tanh())

        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state, stochastic=False):
        mean = self.policy(state)
        dist = Normal(mean, F.softplus(self.std))
        if stochastic:
            return dist.sample().squeeze()
        else:
            return mean.squeeze()

class PPOPolicyContinuous(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOPolicyContinuous, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.ReLU(),
            nn.Linear(100, action_dim),
            nn.Tanh())

        self.vf = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1))

        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state, stochastic=True):
        mean = self.policy(state)
        value = self.vf(state).squeeze()
        dist = Normal(mean, F.softplus(self.std))
        if stochastic:
            action = dist.sample().squeeze()
        else:
            action = mean.squeeze()
        log_prob = dist.log_prob(action).sum(1).squeeze()
        return value, action, log_prob

    def evaluate(self, state, action):
        mean = self.policy(state)
        value = self.vf(state).squeeze()
        dist = Normal(mean, F.softplus(self.std))
        log_prob = dist.log_prob(action).sum(dim=1).squeeze()
        entropy = dist.entropy().sum(dim=-1).squeeze()
        return value, log_prob, entropy

class CMAPPOPolicyContinuous(nn.Module):
    """
    CMA-PPO policy with separate mean and variance networks.
    Actions are sampled from N(mean, var) in pre-tanh space, then tanh is applied.
    """
    def __init__(self, state_dim, action_dim):
        super(CMAPPOPolicyContinuous, self).__init__()
        # Mean network: outputs mean of pre-tanh actions
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.ReLU(),
            nn.Linear(100, action_dim)
        )
        
        # Variance network: outputs log-variance of pre-tanh actions
        self.var_net = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, action_dim)
        )
        
        # Value function (critic) - unchanged from PPO
        self.vf = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1))
    
    def forward(self, state, stochastic=True, return_pre_tanh=False):
        """
        Forward pass: sample action and return value.
        
        Args:
            state: State tensor
            stochastic: Whether to sample stochastically
            return_pre_tanh: If True, return pre-tanh action as well
        
        Returns:
            value, action (tanh-squashed), log_prob
            If return_pre_tanh=True: value, action_tanh, action_pre_tanh, log_prob
        """
        mean = self.mean_net(state)
        log_var = self.var_net(state)
        var = F.softplus(log_var) + 1e-5  # Ensure positive variance
        value = self.vf(state).squeeze()
        
        dist = Normal(mean, var)
        if stochastic:
            action_pre_tanh = dist.sample()
        else:
            action_pre_tanh = mean
        
        # Compute log probability BEFORE squeezing (like PPO does)
        log_prob = dist.log_prob(action_pre_tanh).sum(dim=-1).squeeze()
        
        # Apply tanh squashing and squeeze (like PPO)
        action_tanh = torch.tanh(action_pre_tanh).squeeze()
        action_pre_tanh = action_pre_tanh.squeeze()
        
        # Add tanh correction term (Jacobian of tanh transformation)
        # log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=-1)
        # Actually, for CMA-PPO we work in pre-tanh space, so we don't need the correction
        
        if return_pre_tanh:
            return value, action_tanh, action_pre_tanh, log_prob
        else:
            return value, action_tanh, log_prob
    
    def get_mean_var(self, state):
        """Get mean and variance for given states."""
        mean = self.mean_net(state)
        log_var = self.var_net(state)
        var = F.softplus(log_var) + 1e-5
        return mean, var
    
    def evaluate(self, state, action_pre_tanh):
        """
        Evaluate policy for given states and pre-tanh actions.
        
        Args:
            state: State tensor
            action_pre_tanh: Pre-tanh action tensor
        
        Returns:
            value, log_prob, entropy
        """
        mean = self.mean_net(state)
        log_var = self.var_net(state)
        var = F.softplus(log_var) + 1e-5
        value = self.vf(state).squeeze()
        
        dist = Normal(mean, var)
        log_prob = dist.log_prob(action_pre_tanh).sum(dim=1).squeeze()
        entropy = dist.entropy().sum(dim=-1).squeeze()
        
        return value, log_prob, entropy

def get_policy(policy_type, state_dim=24, action_dim=4):
    """Factory function to create policy networks."""
    policy_type_upper = policy_type.upper()
    if policy_type_upper == 'ES':
        policy = ESPolicyContinuous(state_dim, action_dim)
    elif policy_type_upper in ['CMA_PPO', 'CMAPPO']:
        policy = CMAPPOPolicyContinuous(state_dim, action_dim)
    else:
        policy = PPOPolicyContinuous(state_dim, action_dim)
    
    if torch.cuda.is_available():
        policy = policy.cuda()
    
    return policy

