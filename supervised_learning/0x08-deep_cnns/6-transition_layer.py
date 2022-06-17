#!/usr/bin/env python3
"""
Transition layer
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Transition layer
    """
    initializer = K.initializers.HeNormal()
    batch1 = K.layers.BatchNormalization(axis=3)(X)
    activation1 = K.layers.Activation('relu')(batch1)
    Tr_filters = int(nb_filters * compression)
    Tr_layer = K.layers.Conv2D(filters=Tr_filters,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=initializer)(activation1)
    Tr_pooling = K.layers.AveragePooling2D(pool_size=2,
                                           strides=2)(Tr_layer)
    return Tr_pooling, Tr_filters
