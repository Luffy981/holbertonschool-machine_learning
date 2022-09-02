#!/usr/bin/env python3
"""
Module contains function that performs forward
propagation for a bidirectional RNN.
"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Args:
        bi_cell: Instance of BidirectinalCell for forward propagation.
        X: numpy.ndarray - (t, m, i) Data.
            t: Maximum number of time steps.
            m: Batch size.
            i: Dimensionality of the data.
        h_0: numpy.ndarray - (m, h) Initial hidden state - forward direction.
            h: Dimensionality of the hidden state.
        h_t: numpy.ndarray - (m, h) Initial hidden state - backward direction.
            h: Dimensionality of the hidden state.

    Return: H, Y
        H: numpy.ndarray - Concatenated hidden states.
        Y: numpy.ndarray - Outputs.
    """

    Hf, Hb, h_next, h_prev = [], [], h_t, h_0

    for x, rev_x in zip(X, X[::-1]):
        h_prev = bi_cell.forward(h_prev, x)
        h_next = bi_cell.backward(h_next, rev_x)

        Hf.append(h_prev)
        Hb = [h_next] + Hb

    H = np.concatenate((np.stack(Hf), np.stack(Hb)), axis=-1)

    return H, bi_cell.output(H)
