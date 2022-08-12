#!/usr/bin/env python3
"""
Module contains function that performs the
Baum-Welch algorithm for finding locally optimal
transition and emission probabilities for a Hidden Markov Model.
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


def EM(Observation, Transition, Emission, Initial):
    """
    Expectation Maximization algorithm for updating transition,
    emission, and initial state probabilities to achieve those which
    best generates the observations.

    Args:
        Observations: numpy.ndarray - (T,) Index of the observation.
            T: Number of observations.
        Transition: numpy.ndarray - (M, M) Transition probabilities.
            M: Number of hidden states.
        Emission: numpy.ndarray - (M, N) Emission probabilities.
            N: Number of output states.
        Initial: numpy.ndarray - (M, 1) Starting probabilities.

    Return:
        Emission, Transition, Initial after one update step.
    """

    T = Observation.size
    M, N = Emission.shape
    _, F = forward(Observation, Emission, Transition, Initial)
    _, B = backward(Observation, Emission, Transition, Initial)

    # F[i, j] is the probability of being in hidden state i at time j given
    # the previous observations.

    # B[i, j] is the probability of generating the future observations from
    # hidden state i at time j.

    Xi = np.zeros((T, M, M))

    for t in range(T):
        if t == T - 1:
            op = F[:, t].reshape(M, 1) * Transition # Emission.sum(axis=1)
            Xi[t, :, :] = op.copy()
            break

        op = F[:, t].reshape(M, 1) * Transition * Emission[:, Observation[t+1]]
        op = op * B[:, t+1]
        Xi[t, :, :] = op.copy()

    Xi = Xi / Xi.sum(axis=(1, 2)).reshape(T, 1, 1)

    Transition = (Xi[:T-1, :, :].sum(axis=0) /
                  Xi[:T-1, :, :].sum(axis=(0, 2)).reshape(M, 1))

    for k in range(N):
        idxs = Observation[:T] == k
        Emission[:, k] = Xi[idxs, :, :].sum(axis=(0, 2))/Xi.sum(axis=(0, 2))

    Initial = Xi[0].sum(axis=0)

    return Transition, Emission, Initial.reshape(M, 1)


def baum_welch(Observation, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for finding locally optimal
    transition and emission probabilities for a Hidden Markov Model.

    Args:
        Observations: numpy.ndarray - (T,) Index of the observation.
            T: Number of observations.
        Transition: numpy.ndarray - (M, M) Initialized transition
        probabilities.
            M: Number of hidden states.
        Emission: numpy.ndarray - (M, N) Initialized emission probabilities.
            N: Number of output states.
        Initial: numpy.ndarray - (M, 1) Initialized starting probabilities.
        iterations: Number of times expectation-maximization should
        be performed.

    Return:
        Converged Transition, Emission, or None, None on failure.
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
    if type(iterations) is not int or iterations <= 0:
        return None, None

    for i in range(iterations):
        Transition, Emission, Initial = EM(
            Observation, Transition, Emission, Initial)

    return Transition, Emission
