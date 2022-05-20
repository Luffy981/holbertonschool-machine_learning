#!/usr/bin/env python3
"""RMSprop"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization
    args:
        alpha: is the learning rate
        beta2:  is the RMSProp weight
        epsilon:  is a small number to avoid division by zero
        var: containing the variable to be updated
        grad: containing the gradient of var
        s: is the previous second moment of var
    return:
        the updated variable and the new moment, respectively
    """
    rms = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / ((rms) ** 0.5 + epsilon)
    return var, rms
