#!/usr/bin/env python3
"""forward propagation to neural network"""


import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward propagation"""
    pred = x
    for i in range(len(layer_sizes)):
        pred = create_layer(pred, layer_sizes[i], activations[i])
    return pred
