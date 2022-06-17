#!/usr/bin/env python3
"""
Making Inception Block
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Args:
        A_prev: is the output from the previous layer
        filters: is a tuple or list containing F1, F3R, F3,F5R, F5, FPP
            F1 is the number of filters in the 1x1 convolution
            F3R: is the number of filters in the 1x1 convolution
                 before the 3x3 convolution
            F3: is the number of filters in the 3x3 convolution
            F5R: is the number of filters in the 1x1 convolution
                 before the 5x5 convolution
            F5: is the number of filters in the 5x5 convolution
            FPP: is the number of filters in the 1x1 convolution
                 after the max pooling
    Returns:
        the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    initializer = K.initializers.HeNormal()
    # First cardinal
    conv1 = K.layers.Conv2D(filters=F1,
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer=initializer)(A_prev)
    # second cardinal
    conv2 = K.layers.Conv2D(filters=F3R,
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer=initializer)(A_prev)
    conv3 = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same',
                            kernel_initializer=initializer)(conv2)
    # third cardinal
    conv4 = K.layers.Conv2D(filters=F5R,
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer=initializer)(A_prev)
    conv5 = K.layers.Conv2D(filters=F5,
                            kernel_size=(5, 5),
                            activation='relu',
                            padding='same',
                            kernel_initializer=initializer)(conv4)
    # fourth cardinal
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), padding='same',
                                  strides=(1, 1))(A_prev)
    conv6 = K.layers.Conv2D(filters=FPP,
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer=initializer)(pool1)
    concatenate = K.layers.concatenate(inputs=[conv1, conv3, conv5, conv6])
    return concatenate
