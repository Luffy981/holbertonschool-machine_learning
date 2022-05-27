#!/usr/bin/env python3
"""Forward Propagation with Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Args:
        X is a numpy.ndarray of shape (nx, m) containing the input data
            nx: is the number of input features
            m: is the number of data points
        weights: is a dictionary of the weights and biases of the NN
        L: the number of layers in the network
        keep_prob: is the probability that a node will be kept
    Return:
        dictionary containing the outputs of each layer
        and the dropout mask used on each layer
    """
    # dictionary for activations and masks dropout
    cache = {}
    cache['A0'] = X
    for i in range(1, L):
        inputs = cache['A'+str(i-1)]
        # Ponderate sum
        z = np.dot(weights['W'+str(i)], inputs) + weights['b'+str(i)]
        # Dropout mask
        cache['D' + str(i)] = np.random.binomial(n=1,
                                                 p=keep_prob,
                                                 size=z.shape)
        if i == L - 1:
            # activation function softmax
            cache['A' + str(i)] = softmax(z)
        else:
            # activation function tanh
            factor = np.tanh(z) * cache['D' + str(i)]
            cache['A' + str(i)] = factor / keep_prob
    # dictionary with activations and masks dropout
    return cache


def softmax(z):
    """
    softmax activation
    """
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
