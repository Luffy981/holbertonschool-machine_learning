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

    def evaluate(self, X, Y):
        """Evaluate neuron
        return: Prediction, Cost
        """
        prediction_1, prediction2 = self.forward_prop(X)
        cost = self.cost(Y, prediction2)
        # np.rint: Round elements of the array to the nearest integer.
        prediction2 = np.rint(prediction2).astype(int)
        # print(prediction.shape)
        return (prediction2, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Gradient descent
        Partial derivates of COST FUNCTION respect to Weigth and bias
        1 step
        """
        m = len(Y[0])
        # Derivate cost function
        dz2 = A2 - Y
        # Output layer
        weight_derivative_2 = np.dot(A1, dz2.T) / m
        bias_derivative_2 = np.sum(dz2, axis=1, keepdims=True) / m
        # derivate sigmoid
        d_sigmoid = derivate_sigmoid(A1)
        # derivate cost function
        dz1 = np.dot(self.__W2.T, dz2) * d_sigmoid
        # hidden layer
        weight_derivative_1 = np.dot(X, dz1.T) / m
        bias_derivative_1 = np.sum(dz1, axis=1, keepdims=True) / m
        # Updating weigths and bias hidden layer
        self.__b1 = self.__b1 - (alpha * bias_derivative_1)
        self.__W1 = self.__W1 - (alpha * weight_derivative_1.T)
        # Updating weigths and bias output layer
        self.__b2 = self.__b2 - (alpha * bias_derivative_2)
        self.__W2 = self.__W2 - (alpha * weight_derivative_2.T)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            # calculate predictions
            self.__A1, self.__A2 = self.forward_prop(X)
            # Using gradient to minimize error
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        # Evaluate the training data
        result = self.evaluate(X, Y)
        return result


def sigmoid(number):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-number))


def derivate_sigmoid(z):
    """Sigmoid derivate"""
    return z * (1 - z)
