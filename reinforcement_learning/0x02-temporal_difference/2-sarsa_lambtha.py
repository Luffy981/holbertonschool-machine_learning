#!/usr/bin/env python3
"""
Defines function to perform the SARSA(λ) algorithm
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Epsilon greedy
    """
    # determine exploring-exploiting
    if np.random.uniform(0, 1) < epsilon:
        # exploring
        action = np.random.randint(Q.shape[1])
    else:
        # exploiting
        action = np.argmax(Q[state, :])
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Args:
        env: is the openAI environment instance
        Q: is a numpy.ndarray of shape (s,a) containing the Q table
        lambtha: is the eligibility trace factor
        episodes: is the total number of episodes to train over
        max_steps: is the maximum number of steps per episode
        alpha: is the learning rate
        gamma: is the discount rate
        epsilon: is the initial threshold for epsilon greedy
        min_epsilon: is the minimum value that epsilon should decay to
        epsilon_decay: is the decay rate for updating epsilon between episodes
    Returns:
        Q, the updated Q table
    """
        # set maximum epsilon to the current epsilon before epsilon_decay
    max_epsilon = epsilon
    # Sets the eligibility traces to numpy array of zeros of same shape as Q
    Et = np.zeros((Q.shape))
    # iterate over all episodes
    for ep in range(episodes):
        # set the initial state of each episode to environment reset
        state = env.reset()[0]
        # get the action from epsilon-greedy function
        action = epsilon_greedy(Q, state, epsilon)
        # iterate up to maximum number of steps per episode
        for step in range(max_steps):
            # eligibility traces updated with lambda & gamma
            Et = Et * lambtha * gamma
            # increase Et for current state, action
            Et[state, action] += 1

            # perform the action to get next_state, reward, done, and info
            next_state, reward, done, trunc, info = env.step(action)
            # update the action, using epsilon-greedy again
            next_action = epsilon_greedy(Q, state, epsilon)

            # if the algorithm finds a hole, the reward is updated to -1
            if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
                reward = -1
            # if the algorithm finds the goal, the reward is updated to 1
            if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
                reward = 1
            # BACKWARD VIEW SARSA(λ)
            # calculate delta_t
            # delta_t = R(t + 1) + gamma * Q(St + 1, At + 1) - Q(St, At)
            delta_t = reward + (
                gamma * Q[next_state, next_action]
            ) - Q[state, action]
            # upddate Q table
            # Q(st) = Q(st) + alpha * delta_t * Et(St)
            Q[state, action] = Q[state, action] + (
                alpha * delta_t * Et[state, action])
            # if done, break out of episode
            if done:
                break
            # otherwise, reset state, action and continue
            state = next_state
            action = next_action
        # after each epsiode, update epsilon to decay
        # epsilon will now favor slightly more exploitation than exploration
        epsilon = min_epsilon + (
            (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * ep))
    # when all episodes completed, return updated Q table
    return Q
