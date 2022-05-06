#!/usr/bin/env python3
"""Only just one neuron"""


import numpy as np


class Neuron:
    """Neuron logic"""
    def __init__(self, nx):
        """Initialize neuron"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # np.random.normal(loc=mean,scale=standard_deviation, size=samples)
        # Loc: This parameter defaults to 0
        # Scale: By default, the scale parameter is set to 1.
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weigth to neuron"""
        return self.__W

    @property
    def b(self):
        """Bias to neuron"""
        return self.__b

    @property
    def A(self):
        """prediction for a neuron"""
        return self.__A

    def forward_prop(self, X):
        """Forward propagation function
        X: Input data
        """
        a = sigmoid(np.dot(self.__W, X) + self.__b)
        self.__A = a
        return a


def sigmoid(z):
    """Activation function"""
    return 1 / (1 + np.exp(-z))
