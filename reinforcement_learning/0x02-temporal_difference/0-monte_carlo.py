#!/usr/bin/env python3
"""
Monte Carlo
"""


import numpy as np


def generate_episode(env, policy, max_steps):
    """
    Generates an episode
    """
    # episode = [[state], [rewards]]
    episode = [[], []]
    # The first state comes from resetting the environment
    state = env.reset()[0]
    # Iterate until max number of steps per episode is reached
    for step in range(max_steps):
        # Get action from the current state using policy
        action = policy(state)
        # Perform the action to get next_state, reward, done, truncate, info
        next_state, reward, done, truncate, info = env.step(action)
        # Add current state to the list of episode states
        episode[0].append(state)
        # stop conditions before max_steps reached
        # if the algorithm finds a hole, append reward of -1 & return episode
        if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
            episode[1].append(-1)
            return episode
        # if the algorithm finds the goal, append reward of 1 & return episode
        if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
            episode[1].append(1)
            return episode
        # otherwise, append 0 for no reward & reset current state to nextstate
        episode[1].append(0)
        state = next_state
        # if max steps reached, return the episode
    return episode


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Args:
        env: is the openAI environment instance
        V: is a numpy.ndarray of shape (s,) containing the value estimate
        policy: is a function that takes in a state and returns the next
                action to take
        episodes: is the total number of episodes to train over
        max_steps: is the maximum number of steps per episode
        alpha: is the learning rate
        gamma: is the discount rate
    Returns:
        V, the updated value estimate
    """
    # gamma + gamma ** 1 + gamma ** 2 + gamma ** 3...gamma ** (max_steps - 1)
    discounts = np.array([gamma ** i for i in range(max_steps)])
    for ep in range(episodes):
        # episode = [[state], [rewards]]
        episode = generate_episode(env, policy, max_steps)

        for i in range(len(episode[0])):
            # Recall return -> GT = R_(t+1) + yR_(t+2)+...+ y**(T-1)R_T
            Gt = np.sum(np.array(episode[1][i:]) *
                        np.array(discounts[:len(episode[1][i:])]))
            # Update state -> V(St) = V(St) + alpha * (Gt - V(St))
            V[episode[0][i]] = (V[episode[0][i]] +
                                alpha * (Gt - V[episode[0][i]]))
    return V
