#!/usr/bin/env python3
"""Create a placeholders"""


import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Create Layer"""
    # He et al initialization
    weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    # prev: tensor output of the previous layer
    # n: is the number of nodes in the layer to create layer
    # Initializer function for the weight matrix
    layers = tf.keras.layers.Dense(activation=activation,
                                   units=n,
                                   kernel_initializer=weights,
                                   name='layer')
    return layers(inputs=prev)
