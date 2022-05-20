#!/usr/bin/env python3
"""
batch optimization to SGD
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
       Creates a batch normalization layer for a neural network in tensorflow.
       Args:
         prev is the activated output of the previous layer
         n: number of nodes in the layer to be created
         activation: activation function that should be used on the output
           of the layer
       Returns:
         A tensor of the activated output for the layer.
    """
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layerX = tf.layers.dense(prev, n, kernel_initializer=weights)
    mean, variance = tf.nn.moments(layerX, 0)
    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(
        layerX, mean, variance, beta, gamma, epsilon
    )
    return activation(batch_norm)
