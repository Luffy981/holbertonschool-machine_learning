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
        # Calculating the gradients
        # Derivate forward propagation
        dz = A - Y
        weight_derivative = np.matmul(X, dz.T) / m
        bias_derivative = np.sum(dz) / m
        # Updating weigths and bias
        self.__b = self.__b - (alpha * bias_derivative)
        self.__W = self.__W - (alpha * weight_derivative.T)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=101):
        """Train neuron"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) != int:
                raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")
        # Lists to plot graph
        costs = []
        iterat = []
        for i in range(iterations):
            # calculate prediction
            self.__A = self.forward_prop(X)
            # Using gradient to minimize error
            self.gradient_descent(X, Y, self.__A, alpha)
            if verbose is True and i % step == 0:
                # Calculate current cost
                current_cost = self.cost(Y, self.__A)
                costs.append(current_cost)
                iterat.append(i)
                print("Cost after {} iterations: {}".format(i, current_cost))
        if verbose is True and i % step == 0:
            # Calculate current cost
            current_cost = self.cost(Y, self.__A)
            costs.append(current_cost)
            iterat.append(iterations)
            print("Cost after {} iterations: {}".format(i, current_cost))
        if graph is True:
            plt.plot(iterations, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        # Evaluate the training data
        result = self.evaluate(X, Y)
        return result


def sigmoid(z):
    """Activation function"""
    return 1 / (1 + np.exp(-z))
