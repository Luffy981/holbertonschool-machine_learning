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

    def cost(self, Y, A):
        """
        Cost function CROSS ENTROPY
        Cost=(labels*log(predictions)+(1-labels)*log(1-predictions))/len(labels)
        Params:
            Y: correct labels for the input data
            A: activated output of the neuron for each(prediction)
        """
        # take the error when label = 1
        cost1 = Y * np.log(A)
        # take the error when label = 0
        cost2 = (1 - Y) * np.log(1.0000001 - A)
        # Take the sum of both costs
        total_cost = cost1 + cost2
        # Calculate the number of observations
        # m : number of classes (dog, cat, fish)
        m = len(np.transpose(Y))
        # print(m)
        # Take the average cost
        cost_avg = -total_cost.sum() / m
        return cost_avg

    def evaluate(self, X, Y):
        """Evaluate neuron
        return: Prediction, Cost
        """
        prediction = self.forward_prop(X)
        cost = self.cost(Y, prediction)
        # np.rint: Round elements of the array to the nearest integer.
        prediction = np.rint(prediction).astype(int)
        # print(prediction.shape)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Gradient descent
        Partial derivates of COST FUNCTION respect to Weigth and bias
        1 step
        """
        m = len(Y[0])
        # print(m)
        # Calculatin the gradients
        weight_derivative = (X * (A-Y)).T / m
        # print("WEIGHT ", weight_derivative.shape)
        bias_derivative = np.sum(A - Y) / m
        # print("BIAS ", bias_derivative)
        # Updating weigths and bias
        self.__b = self.__b - (alpha * bias_derivative)
        self.__W = self.__W - (alpha * weight_derivative)


def sigmoid(z):
    """Activation function"""
    return 1 / (1 + np.exp(-z))
