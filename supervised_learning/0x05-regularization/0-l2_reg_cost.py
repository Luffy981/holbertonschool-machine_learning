#!/usr/bin/env python3
"""L2 regularization in Deep Neural Network"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates the cost of a neural network with L2 regularization
    Args:
        cost: is the cost of the network without L2 regularization
        lambtha: is the regularization parameter
        weights: is a dictionary of the weights and biases
        L: is the number of layers in the neural network
        m: is the number of data points used
    Return:
        cost of the network accounting for L2 regularization
    """
    norma = 0
    for key, value in weights.items():
        if key[0] == 'W':
            # nomar2 = sqrt(w²+w²) >>> distance of a vector
            # normal matriz = A.T * A = A*A.T
            norma += np.linalg.norm(value)
    # Calculating ridge regression term
    ridge_reg_term = (lambtha / (2 * m)) * norma
    # Calculating cost with L2
    L2_cost = cost + ridge_reg_term
    return L2_cost
