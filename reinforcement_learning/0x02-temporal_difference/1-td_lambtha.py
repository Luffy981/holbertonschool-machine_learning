#!/usr/bin/env python3
"""
Implementing TD(λ)
"""


import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Args:
        env: is the openAI environment instance
        V: is a numpy.ndarray of shape (s,) containing the value estimate
        policy: is a function that takes in a state and returns the next action to take
        lambtha: is the eligibility trace factor
        episodes: is the total number of episodes to train over
        max_steps: is the maximum number of steps per episode
        alpha: is the learning rate
        gamma: is the discount rate
    Returns:
        V, the updated value estimate
    """
    # set up eligibility traces as a list initialized to 0
    Et = [0 for i in range(env.observation_space.n)]
    for ep in range(episodes):
        # the initial state comes from resetting the environment
        state = env.reset()[0]
        # iterate until done or max number of steps per episode reached
        for step in range(max_steps):
            # list of eligibility traces calculated with lambda & gamma
            # E_t(s) = yλE_(t-1)(S) + 1 (S_t = s)
            Et= list(np.array(Et) * lambtha * gamma)
            # update list by increasing Et at current state
            Et[state] += 1
            # get action from the current state using policy
            action = policy(state)
            # perfomr the action to get next_state, reward, done, trunc, info
            next_state, reward, done, trunc, info = env.step(action)

            # if the algorithm finds a hole, the reward is updated to -1
            if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
                reward = -1
            # if the algorithm finds the goal, the reward is updated to 1
            if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
                reward = 1

            # Backward view TD(λ)
            # delta = reward + (gamma * V[next_state]) - V[state]
            delta_t = reward + gamma * V[next_state] - V[state]
            # V[state] = V[state] + (alpha * delta * elegibility_trace[state])
            V[state] = V[state] + alpha * delta_t * Et[state]
            # Break if done to trigger return
            if done:
                break
            # otherwise, update state to next_state and continue
            state = next_state
            # return V as numpy array when finished

    return np.array(V)

