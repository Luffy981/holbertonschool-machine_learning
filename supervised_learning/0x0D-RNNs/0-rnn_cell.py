#!/usr/bin/env python3
"""
RNN Cell
"""
import numpy as np 

class RNNCell:
    """ Rnn Class """
    def __init__(self, i, h, o):
        """
        Initialize RNN
        Args:
           i: is the dimensionality of the data
           h: is the dimensionality of the hidden state
           o: is the dimensionality of the outputs
        Public Atrributes:
           Wh: Hidden state weights
           bh: Hidden state biases
           Wy: Outputs weights
           by: Outputs biases
        """
        self.Wh = np.random.randn(i+h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """Softmax activation function"""
        max = np.max(x, axis=1, keepdims=True)
        e_x np.exp(x - max)
        sum np.sum(e_x, axis=1, keepdims=True)
        f_x = e_x / sum
        return f_x


    def forward(self, h_prev, x_t):
        """
        Performs forward propagation
        Args:
           xt: (m, i) that contains the data input for the cell
               m: is the batch size of the data
           h_prev: (m, h) containing the previous hidden state
        """
        dot_x = np.dot(x_t, self.Wh[h_prev.shape[1]:, :])
        dot_h = np.dot(h_prev, self.Wh[:h_prev.shape[1], :])
        h_next = np.tanh(dot_x + dot_h + self.bh)
        return h_next, self.softmax(np.dot(h_next, self.Wy) + self.by)

