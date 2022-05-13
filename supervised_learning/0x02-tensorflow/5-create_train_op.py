#!/usr/bin/env python3
"""Gradient descent function"""


import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Gradient descent to neural network"""
    optimize = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    return optimize.minimize(loss=loss)
