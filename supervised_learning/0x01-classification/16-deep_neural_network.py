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
        if type(layers) != list:
            raise TypeError("layers must be a list of positive integers")

        """Layers must be a list of positive integers"""
        """ Using filter to check if any element is negative """
        negative = list(filter(lambda x: x < 0, layers))
        if len(negative > 0):
            raise TypeError("layers must be a list of positive integers")
        """The number of layers in neural network"""
        self.L = len(layers)
        """A dictionary to hold all weigths and biased of the network"""
        self.cache = {}
        self.weights = {}
        for i in range(len(layers)):
            """biases of the network should be initialized to 0’s"""
            if i == 0:
                norm = np.random.randn(layers[i], nx) / np.sqrt(layers[i] / 2)
                self.weights['W' + str(i + 1)] = norm
            else:
                """He et a"""
                factor = layers[i - 1]) / np.sqrt(layers[i] / 2
                nor = np.random.randn(layers[i], factor)
                self.weights['W' + str(i + 1)] = nomr
            col = np.zeros(layers[i]).reshape(layers[i], 1)
            self.weights['b' + str(i + 1)] = col
