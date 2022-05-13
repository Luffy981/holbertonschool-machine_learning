#!/usr/bin/env python3
"""Create a placeholders"""

import tensorflow.compat.v1 as tf

def create_layer(prev, n, activation):
    """ function than can create a tensorflow layer
    ...
    Parameters
    __________
    prev : Tensor
        Previous value of layer
    n : int
        The number of nodes in the layer to create
    activation : function  
        activation function
    ...
    Return
    ______
    layer:
        output of the layer created in tensor
    """
    kernel_initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n, 
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="layer"
    )
    return layer(prev)
