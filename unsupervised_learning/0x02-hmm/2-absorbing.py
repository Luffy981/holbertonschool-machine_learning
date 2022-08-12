#!/usr/bin/env python3
"""
Module contains function for determining
if a markov chain is "absorbing".
"""


import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing.

    Args:
        P: Square 2D numpy.ndarray - (n, n) Standard transition matrix.
        P[i, j]: Probability of transitioning from state i to state j.
        n: Number of states in the markov chain.

    Return:
      True if it is absorbing, or False on failure.
    """

    if type(P) is not np.ndarray or P.ndim != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False

    abs_state = 0
    std = []

    for x, row in enumerate(P):
        if 1 in row:
            std = [x] + std
            abs_state += 1
        else:
            std.append(x)

    if abs_state == 0:
        return False
    if abs_state >= P.shape[0] - 1:
        return True

    standard = (P[std, :])[:, std]

    In = standard[:abs_state, :abs_state]
    Q = standard[abs_state:, abs_state:]
    F = np.linalg.inv(In-Q)

    S = standard[abs_state:, :abs_state]

    absorbtion_probs = np.matmul(F, S)
    QF = np.matmul(F, Q)

    lst = list(range(absorbtion_probs.shape[0]))
    reached = []
    pops = 0
    for x, row in enumerate(absorbtion_probs):
        test = row > 0
        if True in test:
            reached.append(x)
            lst.pop(x-pops)
            pops += 1

    shp = 0
    pops = 0

    while True:
        if shp != len(lst):
            shp = len(lst)
        else:
            return False

        for x in range(len(lst)):
            test = Q[lst[x-pops], reached] > 0
            if True in test:
                reached.append(lst[x-pops])
                lst.pop(x-pops)
                pops += 1

        if len(lst) == 0:
            return True
