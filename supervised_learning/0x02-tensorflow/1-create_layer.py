#!/usr/bin/env python3
"""Creating layers to Neural network"""


import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Create Layer"""
    # He et al initialization
    weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layers = tf.layers.dense(activation=activation,
                             inputs=prev,
                             units=n,
                             kernel_initializer=weights)
    return layers
