#!/usr/bin/env python3
"""Momentun"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
       Creates the training operation for a neural network in tensorflow
       using the gradient descent with momentum optimization algorithm.
       Args:
         loss: loss of the network
         alpha: learning rate
         beta1: momentum weight
       Returns:
         The momentum optimization operation.
    """
    # tensorflow: Optimizer that implements the Momentum algorithm.
    momentun = tf.train.MomentumOptimizer(alpha, beta1)
    return momentun.minimize(loss)
