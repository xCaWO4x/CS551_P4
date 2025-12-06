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

def run_env_CMA_PPO(policy, env_func, max_steps=100, render=False, stochastic=True, reward_only=False, gamma=0.99):
    """
    Environment wrapper for CMA-PPO that returns pre-tanh actions.
    
    Returns:
        states, actions_tanh, actions_pre_tanh, rewards, values, logprobs, returns
    """
    render_mode = 'human' if render else None
    env = env_func(render_mode=render_mode)
    state, info = env.reset()
    done = False
    total_reward = 0
    states = []
    actions_tanh = []
    actions_pre_tanh = []
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
        # CMA-PPO policy returns: value, action_tanh, action_pre_tanh, logprob
        value, action_tanh, action_pre_tanh, logprob = policy.forward(
            to_var(state).unsqueeze(0), stochastic, return_pre_tanh=True
        )
        value, action_tanh, action_pre_tanh, logprob = (
            to_data(value), to_data(action_tanh), to_data(action_pre_tanh), to_data(logprob)
        )
        # Policy already squeezes, so action_tanh should be 1D (action_dim,)
        # But ensure it's the right shape just in case
        if action_tanh.ndim == 0:
            action_tanh = np.array([action_tanh])
        elif action_tanh.ndim > 1:
            action_tanh = action_tanh.squeeze()
        new_state, reward, terminated, truncated, info = env.step(action_tanh)
        done = terminated or truncated
        states.append(state)
        actions_tanh.append(action_tanh)
        actions_pre_tanh.append(action_pre_tanh)
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
    actions_tanh = np.asarray(actions_tanh)
    actions_pre_tanh = np.asarray(actions_pre_tanh)
    rewards = np.asarray(rewards)
    values = np.asarray(values)
    logprobs = np.asarray(logprobs)
    masks = np.asarray(masks)
    if done:
        last_value = 0.0
    else:
        value, _, _, _ = policy.forward(to_var(state).unsqueeze(0), stochastic, return_pre_tanh=True)
        last_value = to_data(value)
        if isinstance(last_value, np.ndarray):
            last_value = float(last_value.item() if last_value.size == 1 else last_value[0])
    
    if reward_only:
        return np.sum(rewards)
    
    # Return raw data - GAE will be computed in the updater
    # For compatibility, we still compute simple returns but they won't be used
    returns = calculate_returns(rewards, masks, last_value, gamma)
    return states, actions_tanh, actions_pre_tanh, rewards, values.squeeze(), logprobs, returns

def calculate_returns(rewards, masks, last_value, gamma=0.99):
    returns = np.zeros(rewards.shape[0] + 1)
    returns[-1] = last_value
    for i in reversed(range(rewards.shape[0])):
        returns[i] = gamma * returns[i+1] * masks[i] + rewards[i]
    returns = np.asarray(returns[:-1])
    return returns

