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

def get_policy(policy_type, state_dim=24, action_dim=4):
    """Factory function to create policy networks."""
    if policy_type == 'ES':
        policy = ESPolicyContinuous(state_dim, action_dim)
    else:
        policy = PPOPolicyContinuous(state_dim, action_dim)
    
    if torch.cuda.is_available():
        policy = policy.cuda()
    
    return policy

