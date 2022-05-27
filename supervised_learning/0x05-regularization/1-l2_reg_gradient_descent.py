#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization in DNN.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network
    using gradient descent with L2 regularization
    Args:
        Y: (classes, m) that contains the correct labels for the data
            classes: is the number of classes
            m: is the number of data points
        weights: is a dictionary of the weights and biases of the DNN
        cache: is a dictionary of the outputs of each layer of the DNN
        alpha: is the learning rate
        lambtha: is the L2 regularization parameter
        L: is the number of layers of the network
    """
    m = Y.shape[1]
    cweights = weights.copy()
    for i in range(L, 0, -1):
        if i == L:
            # Current layer error, derivate cost respect to Z
            curr_layer_err = cache['A'+str(i)] - Y
        else:
            factor = np.dot(cweights['W'+str(i+1)].T, prev_layer_err)
            # Current layer error
            curr_layer_err = factor * derv_tanh(cache['A' + str(i)])
        # derivate cost respect to weight
        derv_cost_w = np.dot(curr_layer_err, cache['A' + str(i-1)].T) / m
        # Regularization with L2
        dw_L2 = derv_cost_w + ((lambtha / m) * cweights['W'+str(i)])
        derv_cost_b = np.sum(curr_layer_err, axis=1, keepdims=True) / m
        # Update weights and bias
        weights['W'+str(i)] = cweights['W'+str(i)] - alpha * dw_L2
        weights['b'+str(i)] = cweights['b'+str(i)] - alpha * derv_cost_b
        # Update layer error
        prev_layer_err = curr_layer_err


def derv_tanh(A):
    """
    Derivate of tanH
    """
    return 1 - (A**2)
