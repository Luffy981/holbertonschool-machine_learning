#!/usr/bin/env python3
"""Create a Layer with Dropout"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Dropout with tensorflow
    Args:
        prev: is a tensor containing the output of the previous layer
        n: is the number of nodes the new layer should contain
        activation: is the activation function that should be used on the layer
        keep_prob: is the probability that a node will be kept
    Return:
        the output of the new layer
    """

    initialice = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                       mode='fan_avg')
    regularizer = tf.keras.layers.Dropout(rate=keep_prob)
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_regularizer=regularizer,
                            kernel_initializer=initialice,
                            name='layer')
    return layer(inputs=prev)
