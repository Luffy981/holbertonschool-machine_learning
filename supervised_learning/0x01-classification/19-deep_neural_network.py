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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            """biases of the network should be initialized to 0’s"""
            if i == 0:
                factor1 = np.random.randn(layers[i], nx)
                factor2 = np.sqrt(2 / nx)
                self.__weights['W' + str(i + 1)] = factor1 * factor2
            else:
                """He et a"""
                factor1 = np.random.randn(layers[i], layers[i - 1])
                factor2 = np.sqrt(2 / layers[i - 1])
                self.__weights['W' + str(i + 1)] = factor1 * factor2
            zeros = np.zeros(layers[i])
            self.__weights['b' + str(i + 1)] = zeros.reshape(layers[i], 1)

    @property
    def cache(self):
        """ A dictionary to hold all intermediary values of the network"""
        return self.__cache

    @property
    def L(self):
        """The number of layers in the neural network"""
        return self.__L

    @property
    def weights(self):
        """ A dictionary to hold all weights and biased of the network"""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache['A0'] = X
        for i in range(self.__L):
            w_layer = self.__weights['W' + str(i + 1)]
            b_layer = self.__weights['b' + str(i + 1)]
            activation = sigmoid(np.dot(w_layer, X) + b_layer)
            X = activation
            self.__cache['A' + str(i + 1)] = activation
        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Cost function CROSS ENTROPY
        Cost=(labels*log(predictions)+(1-labels)*log(1-predictions))/len(labels)
        Params:
            Y: correct labels for the input data
            A: activated output of the neuron for each(prediction)
        """
        """take the error when label = 1"""
        cost1 = Y * np.log(A)
        """take the error when label = 0"""
        cost2 = (1 - Y) * np.log(1.0000001 - A)
        """Take the sum of both costs"""
        total_cost = cost1 + cost2
        """Calculate the number of observations"""
        """m : number of classes (dog, cat, fish)"""
        m = len(np.transpose(Y))
        """print(m)"""
        """Take the average cost"""
        cost_avg = -total_cost.sum() / m
        return cost_avg


def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))
