import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math

def get_policy(args, env):
    if args.alg == 'ES':
        # add the model on top of the convolutional base
        policy = ESPolicyContinuous(24, 4)
    else:
        policy = PPOPolicyContinuous(24, 4)
    if torch.cuda.is_available():
        policy = policy.cuda()
    return policy

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
            return dist.sample().squeeze() # depends on action space type (box or discrete)
        else:
            return mean.squeeze() # depends on action space type (box or discrete)

class PPOPolicyContinuous(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOPolicyContinuous, self).__init__()
        #body = nn.Sequential(
        #                       nn.Linear(state_dim, 100),
        #                       nn.ReLU(),
        #                       nn.Linear(100, 100))

        self.policy = nn.Sequential(
                                nn.Linear(state_dim, 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                #nn.Linear(state_dim, 100),
                                nn.Tanh(),
                                nn.ReLU(),
                                nn.Linear(100, action_dim),
                                nn.Tanh())
                                #nn.Linear(100, action_dim))#,
                                #nn.Softmax(dim=1))

        self.vf = nn.Sequential(
                                nn.Linear(state_dim, 100),
                                nn.ReLU(),
                                nn.Linear(100, 100),
                                #nn.Linear(state_dim, 100),
                                #nn.Tanh(),
                                #nn.Linear(100, 100),
                                #nn.Tanh(),
                                nn.ReLU(),
                                nn.Linear(100, 1))

        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state, stochastic=True):
        mean = self.policy(state)
        value = self.vf(state).squeeze()
        dist = Normal(mean, F.softplus(self.std))
        if stochastic:
            action = dist.sample().squeeze() # depends on action space type (box or discrete)
        else:
            action = mean.squeeze() # depends on action space type (box or discrete)
        log_prob = dist.log_prob(action).sum(1).squeeze()
        return value, action, log_prob

    def evaluate(self, state, action):
        mean = self.policy(state)
        value = self.vf(state).squeeze()
        dist = Normal(mean, F.softplus(self.std))
        log_prob = dist.log_prob(action).sum(dim=1).squeeze()
        entropy = dist.entropy().sum(dim=-1).squeeze()
        return value, log_prob, entropy

