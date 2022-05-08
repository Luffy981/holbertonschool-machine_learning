#!/usr/bin/env python3
"""deep neural network"""


import numpy as np


class DeepNeuralNetwork:
    """making a deep neural network"""
    def __init__(self, nx, layers):
        """initialiaze deep neural network"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        """ Using filter to check if any element is negative """
        negative = list(filter(lambda x: x <= 0, layers))
        if len(negative) > 0:
            raise TypeError("layers must be a list of positive integers")
        """The number of layers in neural network"""
        self.L = len(layers)
        """A dictionary to hold all weigths and biased of the network"""
        self.cache = {}
        self.weights = {}
        for i in range(len(layers)):
            """biases of the network should be initialized to 0’s"""
            if i == 0:
                factor1 = np.random.randn(layers[i], nx)
                factor2 = np.sqrt(2 / nx)
                self.weights['W' + str(i + 1)] = factor1 * factor2
            else:
                """He et a"""
                factor1 = np.random.randn(layers[i], layers[i - 1])
                factor2 = np.sqrt(2 / layers[i - 1])
                self.weights['W' + str(i + 1)] = factor1 * factor2
            zeros = np.zeros(layers[i])
            self.weights['b' + str(i + 1)] = zeros.reshape(layers[i], 1)
