#!/usr/bin/env python3
"""
Create a Layer with L2 Regularization
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization
    Args:
        prev: is a tensor containing the output of the previous layer
        n: is the number of nodes the new layer should contain
        activation: is the activation function
        lambtha: is the L2 regularization parameter
    Returns:
        the output of the new layer


    """
    # Regularizer L2
    regularizer = tf.keras.regularizers.L2(l2=lambtha)
    # initializer He et al
    weights = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                    mode='fan_avg')
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=weights,
                            kernel_regularizer=regularizer,
                            name='layer')
    return layer(inputs=prev)
