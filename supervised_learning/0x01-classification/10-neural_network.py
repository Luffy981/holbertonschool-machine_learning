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
        self.__W1 = np.random.normal(size=(nodes, nx))
        """The bias for the hidden layer"""
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        """The activated output for the hidden layer"""
        self.__A1 = 0
        """The weights vector for the output neuron"""
        self.__W2 = np.random.normal(size=(1, nodes))
        """The bias for the output neuron"""
        self.__b2 = 0
        """The activated output for the output neuron (prediction)"""
        self.__A2 = 0

    @property
    def W1(self):
        """weights vector for the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """bias for the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Prediction for the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Weights vector for the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """bias for the output neuron """
        return self.__b2

    @property
    def A2(self):
        """prediction for the output neuron"""
        return self.__A2

    def forward_prop(self, X):
        """Forward propagation"""
        """Ponderate weights and data"""
        sump1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = sigmoid(sump1)
        sump2 = np.dot(self.__W2, self.__A1) + self.__b2
        """using activation function"""
        A2 = sigmoid(sump2)
        """Updating predictions for hidden layer and output layer"""
        self.__A2 = A2
        return (self.__A1, self.__A2)


def sigmoid(number):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-number))
