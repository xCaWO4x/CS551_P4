import numpy as np
import copy
import gymnasium as gym
import torch
import time
from utils import to_var, to_data

# weights is first argument because of threadpool
def run_env_ES(weights, policy, env_func, render=False, stochastic=False):
    render_mode = 'human' if render else None
    env = env_func(render_mode=render_mode)
    cloned_policy = copy.deepcopy(policy)
    for i, weight in enumerate(cloned_policy.parameters()):
        try:
            weight.data.copy_(weights[i])
        except:
            weight.data.copy_(weights[i].data)
    state, info = env.reset()
    done = False
    total_reward = 0
    step  = 0
    while not done:
        if render:
            time.sleep(0.05)
        action = cloned_policy.forward(to_var(state).unsqueeze(0), stochastic)
        action = to_data(action)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
    env.close()
    return total_reward

def run_env_PPO(policy, env_func, max_steps=100, render=False, stochastic=True, reward_only=False, gamma=0.99):
    render_mode = 'human' if render else None
    env = env_func(render_mode=render_mode)
    state, info = env.reset()
    done = False
    total_reward = 0
    states = []
    actions = []
    rewards = []
    values = []
    logprobs = []
    masks = []
    step = 0
    while True:
        if (step == max_steps) and (not reward_only):
            break
        if render:
            time.sleep(0.05)
        value, action, logprob = policy.forward(to_var(state).unsqueeze(0), stochastic)
        value, action, logprob = to_data(value), to_data(action), to_data(logprob)
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        logprobs.append(logprob)
        masks.append(1-done)
        total_reward += reward
        state = new_state
        if done:
            if reward_only:
                env.close()
                return total_reward
            else:
                state, info = env.reset()
        step += 1
    env.close()
    states = np.asarray(states)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    values = np.asarray(values)
    logprobs = np.asarray(logprobs)
    masks = np.asarray(masks)
    if done:
        last_value = 0.0
    else:
        last_value, _, _ = policy.forward(to_var(state).unsqueeze(0), stochastic)
        last_value = to_data(last_value)
    returns = calculate_returns(rewards, masks, last_value, gamma)
    if reward_only:
        return np.sum(rewards)
    return states, actions, rewards, values.squeeze(), logprobs, returns

def calculate_returns(rewards, masks, last_value, gamma=0.99):
    returns = np.zeros(rewards.shape[0] + 1)
    returns[-1] = last_value
    for i in reversed(range(rewards.shape[0])):
        returns[i] = gamma * returns[i+1] * masks[i] + rewards[i]
    returns = np.asarray(returns[:-1])
    return returns

