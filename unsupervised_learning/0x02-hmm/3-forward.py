#!/usr/bin/env python3
"""
Module contains function for computing forward
algorithm for forward HHM algorithm.
"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model.

    Args:
        Observation: numpy.ndarray - (T,) that contains the index of
        the observation.
            - T: Number of observations.
        Emission: numpy.ndarray - (N, M) Emission probability of a specific
        observation given a hidden state.
            - Emission[i, j]: Probability of observing j given the hidden
            state i.
            - N: Number of hidden states.
            - M: Number of all possible observations.
        Transition: numpy.ndarray - (N, N) Transition probabilities.
            - Transition[i, j]: Probability of transitioning from the hidden
            state i to j.
        Initial: numpy.ndarray - (N, 1) Probability of starting in a particular
        hidden state.

    Return: P, F, or None, None on failure
        P: Likelihood of the observations given the model.
        F: numpy.ndarray of shape (N, T) containing the forward path
        probabilities.
            - F[i, j]: Probability of being in hidden state i at time j given
            the previous observations.
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

    n, m = Emission.shape

    # N x 1 (time step 0)
    F = (Initial.T * Emission[:, Observation[0]]).T

    for t in range(1, Observation.shape[0]):
        rt = F[:, t-1].T * Transition.T.reshape(n, 1, n)
        rt = rt * Emission[:, Observation[t]].reshape(n, 1, 1)
        F = np.concatenate((F, rt.sum(-1)), axis=1)

    return np.sum(F[:, -1]), F
