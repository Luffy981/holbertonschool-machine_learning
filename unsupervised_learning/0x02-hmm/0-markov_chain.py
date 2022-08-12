#!/usr/bin/env python3
"""
Module contains function representing a
markov chain.
"""


import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a
    particular state after a specified number of iterations.

    Args:
        P: Square 2D numpy.ndarray - (n, n) representing the
        transition matrix.
            - P[i, j] is the probability of transitioning from state
            i to state j.
        n: Number of states in the markov chain.
        s: numpy.ndarray - (1, n) representing the probability of starting
        in each state.
        t: Number of iterations that the markov chain has been through.

    Returns:
        numpy.ndarray of shape (1, n) Probability of being in a specific
        state after t iterations, or None on failure.
    """

    if type(P) is not np.ndarray or type(t) is not int:
        return None
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return None
    if type(s) is not np.ndarray or t <= 0:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None

    current_state = s.copy()

    for step in range(t):
        current_state = np.matmul(current_state, P)

    return current_state
