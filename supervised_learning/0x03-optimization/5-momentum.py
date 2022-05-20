#!/usr/bin/env python3
"""Momentun"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
         gradient descent with momentum optimization
         args:
            alpha: is the learning rate
            beta1: is the momentum weight
            var: containing the variable to be updated
            grad: containing the gradient of var
            v: is the previous first moment of var
        returns:
            variable and the new moment, respectively
    """
    vt = beta1 * v + ((1 - beta1) * grad)
    var = var - alpha * vt
    return var, vt
