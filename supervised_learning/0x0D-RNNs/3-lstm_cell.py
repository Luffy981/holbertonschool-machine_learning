#!/usr/bin/env python3
"""
Module contains class LSTMCell that represents an LSTM
"""


import numpy as np


class LSTMCell():
    """
    Represents a Long Short Term Memory (LSTM) unit
    for a Recurrent Neural Network.
    """

    def __init__(self, i, h, o):
        """
        Class constructor.

        Args:
            i: Dimensionality of the data.
            h: Dimensionality of the hidden state.
            o: Dimensionality of the outputs.

        Public Attributes:
            Wf: Forget gate weights.
            bf: Forget gate biases.
            Wy: Output weights.
            by: Output biases.
            Wu: Update gate weights.
            bu: Update gate biases.
            Wc: Intermediate cell state weights.
            bc: Intermediate cell state biases.
            Wo: Output gate weights.
            bo: Output gate biases.
        """

        self.Wf = np.random.randn(i+h, h)
        self.Wu = np.random.randn(i+h, h)
        self.Wc = np.random.randn(i+h, h)
        self.Wo = np.random.randn(i+h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.bc = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bo = np.zeros((1, h))

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

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propogation for 1 time step.

        Args:
            x_t: numpy.ndarray - (m, i) Data input for the cell.
                m: Batch size for the data.
            h_prev: numpy.ndarray - (m, h) Previous hidden state.
            c_prev: numpy.ndarray - (m, h) Previous cell state.

        Return: h_next, y
            h_next: Next hidden state.
            c_next: Next cell state.
            y: Output of the cell.
        """

        H_SHAPE = h_prev.shape[1]

        # Update Gate
        dot_xi = np.dot(x_t, self.Wu[H_SHAPE:, :])
        dot_hi = np.dot(h_prev, self.Wu[:H_SHAPE, :])
        It = self.sigmoid(dot_xi + dot_hi + self.bu)

        # Forget Gate
        dot_xf = np.dot(x_t, self.Wf[H_SHAPE:, :])
        dot_hf = np.dot(h_prev, self.Wf[:H_SHAPE, :])
        F = self.sigmoid(dot_xf + dot_hf + self.bf)

        # Output Gate
        dot_xo = np.dot(x_t, self.Wo[H_SHAPE:, :])
        dot_ho = np.dot(h_prev, self.Wo[:H_SHAPE, :])
        Ot = self.sigmoid(dot_xo + dot_ho + self.bo)

        # Candidate Memory
        dot_xc = np.dot(x_t, self.Wc[H_SHAPE:, :])
        dot_hc = np.dot(h_prev, self.Wc[:H_SHAPE, :])
        C = np.tanh(dot_xc + dot_hc + self.bc)

        c_next = F * c_prev + It * C

        h_next = Ot * np.tanh(c_next)

        return h_next, c_next, self.softmax(np.dot(h_next, self.Wy) + self.by)
