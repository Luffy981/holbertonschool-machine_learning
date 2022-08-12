#!/usr/bin/env python3
"""
Module contains function performing the
Viterbi algorithm for a Hidden Markov Model.
"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden
    markov model.

    Args:
        Observation: numpy.ndarray - (T,) that contains the index of the
        observation.
            T: Number of observations.
        Emission: numpy.ndarray - (N, M) Emission probabilities of a specific
        observation given a hidden state.
            Emission[i, j]: robability of observing j given the hidden state i.
            N: Number of hidden states.
            M: Number of all possible observations.
        Transition: 2D numpy.ndarray - (N, N) Transition probabilities.
            Transition[i, j]: Probability of transitioning from the hidden
            state i to j.
        Initial: numpy.ndarray - (N, 1) Probability of starting in a particular
        hidden state.

    Return: path, P, or None and None
        path: List of length T containing the most likely sequence of hidden
        states.
        P: Probability of obtaining the path sequence.
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

    N, _ = Emission.shape
    T = Observation.size

    seq_probs = Initial * Emission[:, Observation[0]][..., np.newaxis]
    buff = np.zeros((N, T))

    for t in range(1, T):
        mat = (Emission[:, Observation[t]] * Transition.reshape(N, 1, N))
        mat = (mat.reshape(N, N) * seq_probs[:, t-1].reshape(N, 1))

        mx = np.max(mat, axis=0).reshape(N, 1)
        seq_probs = np.concatenate((seq_probs, mx), axis=1)
        buff[:, t] = np.argmax(mat, axis=0).T

    P = np.max(seq_probs[:, T-1])
    link = np.argmax(seq_probs[:, T-1])
    path = [link]

    for t in range(T - 1, 0, -1):
        idx = int(buff[link, t])
        path.append(idx)
        link = idx

    return path[::-1], P
