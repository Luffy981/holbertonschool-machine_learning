#!/usr/bin/env python3
"""
Dense network
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Dense network
    """
    initializer = K.initializers.HeNormal()
    input_layer = K.Input(shape=(224, 224, 3))
    nb_filters = growth_rate * 2
    batch1 = K.layers.BatchNormalization(axis=3)(input_layer)
    activation1 = K.layers.Activation('relu')(batch1)
    conv1 = K.layers.Conv2D(filters=nb_filters,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer)(activation1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(conv1)
    X1, nb_filters = dense_block(pool1, nb_filters, growth_rate, layers=6)
    Tr_layer1, nb_filters = transition_layer(X1, nb_filters, compression)

    X2, nb_filters = dense_block(Tr_layer1, nb_filters, growth_rate, 12)
    Tr_layer2, nb_filters = transition_layer(X2, nb_filters, compression)

    X3, nb_filters = dense_block(Tr_layer2, nb_filters, growth_rate, 24)
    Tr_layer3, nb_filters = transition_layer(X3, nb_filters, compression)

    X4, nb_filters = dense_block(Tr_layer3, nb_filters, growth_rate, 16)
    pool2 = K.layers.AveragePooling2D(pool_size=7)(X4)
    fc1 = K.layers.Dense(units=1000, activation='softmax',
                         kernel_initializer=initializer)(pool2)
    model = K.models.Model(inputs=input_layer, outputs=fc1)
    return model
