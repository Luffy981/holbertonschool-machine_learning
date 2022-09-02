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

    def backward(self, h_next, x_t):
        """
        Calculates hidden state in backward direction for 1 time step.

        Args:
            x_t: numpy.ndarray - (m, i) Data input for the cell.
                m: Batch size for the data.
            h_next: numpy.ndarray - (m, h) Next hidden state.

        Return: h_next
            h_prev: Previous hidden state.
        """

        dot_x = np.dot(x_t, self.Whb[h_next.shape[1]:, :])
        dot_h = np.dot(h_next, self.Whb[:h_next.shape[1], :])
        return np.tanh(dot_x + dot_h + self.bhb)

    def output(self, H):
        """
        Calculates all outputs for the bidirectional RNN.

        Args:
            H: numpy.ndarray - (t, m, 2 * h) Concatenated hidden states
            from both directions, excluding their initialized states.
                t: Number of time steps.
                m: Batch size for the data.
                h: Dimensionality of hidden states.

        Return: Y
            Y: numpy.ndarray - (t, m, y) Outputs.
        """

        Y = [self.softmax(np.dot(h, self.Wy) + self.by) for h in H]
        return np.stack(Y)
