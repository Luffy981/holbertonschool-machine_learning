#!/usr/bin/env python3
"""Making neural network"""


import numpy as np


class NeuralNetwork:
    """making Neural network"""
    def __init__(self, nx, nodes):
        """Initialize neural network"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        """The weights vector for the hidden layer"""
        self.W1 = np.random.normal(size=(nodes, nx))
        """The bias for the hidden layer"""
        self.b1 = np.zeros(nodes).reshape(nodes, 1)
        """The activated output for the hidden layer"""
        self.A1 = 0
        """The weights vector for the output neuron"""
        self.W2 = np.random.normal(size=(1, nodes))
        """The bias for the output neuron"""
        self.b2 = 0
        """The activated output for the output neuron (prediction)"""
        self.A2 = 0
