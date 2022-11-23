#!/usr/bin/env python3
"""
Function to compute Monte Carlo policy gradient based on
state and weight matrices
"""


import numpy as np


def policy(matrix, weight):
    """
    Function that computes to policy with a weight of a matrix.
    """
    # for each column of weights, sum (matrix[i] * weight[i]) using dot product
    dot_product = matrix.dot(weight)
    # find the exponent of the calculated dot product
    exp = np.exp(dot_product)
    # policy is exp / sum(exp)
    policy = exp / np.sum(exp)
    return policy


def policy_gradient(state, weight):
    """
    state: matrix representing the current observation of the environment
    weight: matrix of random weight
    Return: the action and the gradient (in this order)
    """
    # first calculate policy using the policy function above
    Policy = policy(state, weight)
    # get action from policy
    action = np.random.choice(len(Policy[0]), p=Policy[0])
    # reshape single feature from policy
    s = Policy.reshape(-1, 1)
    # apply softmax function to s and access value at action
    softmax = (np.diagflat(s) - np.dot(s, s.T))[action, :]
    # calculate the dlog as softmax / policy at action
    dlog = softmax / Policy[0, action]
    # find gradient from input state matrix using dlog
    gradient = state.T.dot(dlog[None, :])
    # return action and the policy gradient
    return action, gradient
