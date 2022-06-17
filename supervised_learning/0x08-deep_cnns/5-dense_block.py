#!/usr/bin/env python3
"""
Dense block
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Dense block
    """
    initializer = K.initializers.HeNormal()
    for layer in range(layers):
        batch1 = K.layers.BatchNormalization(axis=3)(X)
        activation1 = K.layers.Activation('relu')(batch1)
        conv1 = K.layers.Conv2D(filters=(4 * growth_rate),
                                kernel_size=(1, 1),
                                padding='same',
                                kernel_initializer=initializer)(activation1)
        batch2 = K.layers.BatchNormalization(axis=3)(conv1)
        activation2 = K.layers.Activation('relu')(batch2)
        conv2 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3),
                                padding='same',
                                kernel_initializer=initializer)(activation2)
        X = K.layers.concatenate([X, conv2])
        nb_filters += growth_rate
    return X, nb_filters
