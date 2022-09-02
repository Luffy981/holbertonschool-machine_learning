#!/usr/bin/env python3
"""
Module contains function that performs forward
propagation for a deep RNN.
"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells: List of RNNCell instances of length l.
            l: Number of layers.
        X: numpy.ndarray - (t, m, i) Data.
            t: Maximum number of time steps.
            m: Batch size.
            i: Dimensionality of the data.
        h_0: numpy.ndarray - (l, m, h) Initial hidden state.
            h: Dimensionality of the hidden state.

    Return: H, Y
        H: numpy.ndarray - Hidden states.
        Y: numpy.ndarray - Outputs.
    """

    LAYERS = len(rnn_cells)
    H, Y = [h_0], []
    temp_h = h_0.copy()

    for x in X:
        temp_h[0], _ = rnn_cells[0].forward(temp_h[0], x)

        for i in range(LAYERS - 1):
            temp_h[1+i], out = rnn_cells[1+i].forward(temp_h[1+i], temp_h[i])

            if i == LAYERS - 2:
                Y.append(out)

        H.append(temp_h.copy())

    return np.stack(H), np.stack(Y)
