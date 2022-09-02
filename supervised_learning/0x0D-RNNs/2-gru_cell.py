#!/usr/bin/env python3
"""
Module contains class GRUCell, representing a Gated
Recurrent Unit for an RNN.
"""


import numpy as np


class GRUCell():
    """Represents a Gated Recurrent Unit"""

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
            Wz: Update gate weights.
            bz: Update gate biases.
            Wr: Reset gate weights.
            br: Reset gate biases.
        """

        self.Wz = np.random.randn(i+h, h)
        self.Wr = np.random.randn(i+h, h)
        self.Wh = np.random.randn(i+h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))

    @staticmethod
    def softmax(x):
        """softmax activation function"""

        max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - max)
        sum = np.sum(e_x, axis=1, keepdims=True)
        f_x = e_x / sum
        return f_x

    @staticmethod
    def sigmoid(x):
        """sigmoid function"""
        return 1/(1 + np.exp(-x))

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

        H_SHAPE = h_prev.shape[1]

        # Reset Gate
        dot_xr = np.dot(x_t, self.Wr[H_SHAPE:, :])
        dot_hr = np.dot(h_prev, self.Wr[:H_SHAPE, :])
        R = self.sigmoid(dot_xr + dot_hr + self.br)

        # Update Gate
        dot_xz = np.dot(x_t, self.Wz[H_SHAPE:, :])
        dot_hz = np.dot(h_prev, self.Wz[:H_SHAPE, :])
        Z = self.sigmoid(dot_xz + dot_hz + self.bz)

        # Candidate Hidden State
        dot_xh = np.dot(x_t, self.Wh[H_SHAPE:, :])
        dot_Rhh = np.dot(h_prev * R, self.Wh[:H_SHAPE, :])
        C = np.tanh(dot_xh + dot_Rhh + self.bh)

        h_next = (1 - Z) * h_prev + Z * C

        return h_next, self.softmax(np.dot(h_next, self.Wy) + self.by)
