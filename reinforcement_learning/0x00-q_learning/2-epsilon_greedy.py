#!/usr/bin/env python3
"""
uses epsilon-greedy to determine the next action
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Args:
        Q: numpy.ndarray containing the q-table
        state:  is the current state
        epsilon: is the epsilon to use for the calculation
    """
    exploration_rate_threshold = np.random.uniform(0, 1)
    if exploration_rate_threshold > epsilon:
        # Exploiting
        action = np.argmax(Q[state, :])
    else:
        # Expploring
        action = np.random.randint(Q.shape[1])
    return action
