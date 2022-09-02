#!/usr/bin/env python3
"""
Module contains class BidirectionalCell that
represents a bidirectional cell of an RNN.
"""


import numpy as np


class BidirectionalCell():
    """Represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """
        Class constructor.

        Args:
            i: Dimensionality of the data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the outputs.

        Public Attributes:
            Whf: Hidden state weights for forward direction.
            bhf: Hidden state biases for forward direction.
            Whb: Hidden state weights for backward direction.
            bhb: Hidden state biases for backward direction.
            Wy: Output weights.
            by: Output biases.
        """

        self.Whf = np.random.randn(i+h, h)
        self.Whb = np.random.randn(i+h, h)
        self.Wy = np.random.randn(2*h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates hidden state in forward direction for 1 time step.

        Args:
            x_t: numpy.ndarray - (m, i) Data input for the cell.
                m: Batch size for the data.
            h_prev: numpy.ndarray - (m, h) Previous hidden state.

        Return: h_next
            h_next: Next hidden state.
        """

        dot_x = np.dot(x_t, self.Whf[h_prev.shape[1]:, :])
        dot_h = np.dot(h_prev, self.Whf[:h_prev.shape[1], :])
        return np.tanh(dot_x + dot_h + self.bhf)
