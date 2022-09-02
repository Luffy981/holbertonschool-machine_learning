#!/usr/bin/env python3
"""
Module contains function for iterating through
RNN forward propogation.
"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: Instance of RNNCell used for the forward propagation.
        X: numpy.ndarray - (t, m, i) Data to use.
            t: Maximum number of time steps.
            m: Batch size.
            i: Dimensionality of the data.
        h_0: numpy.ndarray - (m, h) Initial hidden state.
            h: Dimensionality of the hidden state.
    Return: H, Y
        H: numpy.ndarray - All of the hidden states.
        Y: numpy.ndarray - All of the outputs.
    """

    H, Y, h_next = [h_0], [], h_0

    for x in X:
        h_next, y = rnn_cell.forward(h_next, x)
        H.append(h_next), Y.append(y)

    return np.stack(H), np.stack(Y)
