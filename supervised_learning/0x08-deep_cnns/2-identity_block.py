#!/usr/bin/env python3
"""
Identity Block
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    identity block
    """
    initializer = K.initializers.HeNormal()
    F11, F3, F12 = filters
    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=initializer)(A_prev)
    bach_norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    activation1 = K.layers.Activation('relu')(bach_norm1)
    conv2 = K.layers.Conv2D(filters=F3,
                            padding='same',
                            kernel_size=(3, 3),
                            kernel_initializer=initializer)(activation1)
    bach_norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    activation2 = K.layers.Activation('relu')(bach_norm2)
    conv3 = K.layers.Conv2D(filters=F12,
                            padding='same',
                            kernel_size=(1, 1),
                            kernel_initializer=initializer)(activation2)
    bach_norm3 = K.layers.BatchNormalization(axis=3)(conv3)

    add_layer = K.layers.Add()([bach_norm3, A_prev])
    activation3 = K.layers.Activation('relu')(add_layer)
    return activation3
