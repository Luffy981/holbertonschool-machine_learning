#!/usr/bin/env python3
"""Builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Args:
        nx: is the number of input features to the network
        layers: is a list containing the number of nodes in each layer
        of the network
        activations: is a list containing the activation functions used
        for each layer of the network
        lambtha: is the L2 regularization parameter
        keep_prob: is the probability that a node will be kept for dropout
    Return:
        the keras model
    """
    # define the keras model
    model = K.Sequential()
    regularizer = K.regularizers.L2(l2=lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer,
                                     input_shape=(nx, )))
        else:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(rate=(1 - keep_prob)))
    model.compile(optimizer='adam')
    return model
