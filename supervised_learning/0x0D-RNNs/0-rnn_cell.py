#!/usr/bin/env python3
"""
Module contains RNNCell class that represents a cell of
a simple Recurrent Neural Network.
"""


import numpy as np


class RNNCell():
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """
        Class constructor.

        Args:
            i: Dimensionality of the data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the outputs.

        Public Attributes:
            Wh: Hidden state weights.
            bh: Hidden state biases.
            Wy: Output weights.
            by: Output biases.
        """

        self.Wh = np.random.randn(i+h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """softmax activation function"""

        max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - max)
        sum = np.sum(e_x, axis=1, keepdims=True)
        f_x = e_x / sum
        return f_x

    def forward(self, h_prev, x_t):
        """
        Performs forward propogation for 1 time step.

        Args:
            x_t: numpy.ndarray - (m, i) Data input for the cell.
                m: Batch size for the data.
            h_prev: numpy.ndarray - (m, h) Previous hidden state.

        Return: h_next, y
            h_next: Next hidden state.
            y: Output of the cell.
        """

        dot_x = np.dot(x_t, self.Wh[h_prev.shape[1]:, :])
        dot_h = np.dot(h_prev, self.Wh[:h_prev.shape[1], :])
        h_next = np.tanh(dot_x + dot_h + self.bh)
        return h_next, self.softmax(np.dot(h_next, self.Wy) + self.by)
