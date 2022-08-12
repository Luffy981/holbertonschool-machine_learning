#!/usr/bin/env python3
"""
Module contains function for performing the
backward algorithm for a hidden markov model.
"""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model.

    Args:
        Observation:numpy.ndarray - (T,) that contains the index of the
        observation.
            T: Number of observations.
        Emission: numpy.ndarray - (N, M) Emission probabilities of a specific
        observation given a hidden state.
            Emission[i, j]: Probability of observing j given the hidden
            state i.
            N: Number of hidden states.
            M: Number of all possible observations.
        Transition: 2D numpy.ndarray - (N, N) Transition probabilities.
            Transition[i, j] Probability of transitioning from the hidden
            state i to j.
        Initial a numpy.ndarray - (N, 1) Probability of starting in a
        particular hidden state.

    Return: P, B, or None, None
        P: Likelihood of the observations given the model.
        B: numpy.ndarray - (N, T) Backward path probabilities.
        B[i, j]: Probability of generating the future observations from
        hidden state i at time j.
    """

    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if Observation.shape[0] == 0:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial) != Transition.shape[0]:
        return None, None

    N, M = Emission.shape
    T = Observation.size

    B = np.ones((N, T), dtype="float")

    for t in range(T-2, -1, -1):
        mat = (Emission[:, Observation[t+1]] * Transition.reshape(N, 1, N))
        mat = (B[:, t+1] * mat).reshape(N, N).sum(axis=1)

        B[:, t] = mat

    P = (Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P.sum(axis=1)[0], B
