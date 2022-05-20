#!/usr/bin/env python3
"""
Adam optimization algorithm
"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
       Updates a variable in place using the Adam optimization algorithm.
       Args:
         alpha: learning rate
         beta1: weight used for the first moment
         beta2: weight used for the second moment
         epsilon: small number to avoid division by zero
         var: numpy.ndarray containing the variable to be updated
         grad: numpy.ndarray containing the gradient of var
         v: previous first moment of var
         s: previous second moment of var
         t: time step used for bias correction
       Returns:
         The updated variable, the new first moment, and the new second moment,
         respectively.
    """
    vdw = (beta1*v) + ((1-beta1)*grad)
    vdw_correct = vdw/(1-(beta1**t))
    sdw = (beta2*s) + (1-beta2)*(grad**2)
    sdw_correct = sdw/(1-(beta2**t))
    sdw_correct_sqrt = (sdw_correct**(1/2.0))+epsilon
    return var - (alpha*(vdw_correct/sdw_correct_sqrt)), vdw, sdw
