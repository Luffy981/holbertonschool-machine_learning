#!/usr/bin/env python3
"""
Making Inception Block
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds the inception network as described in Going Deeper
    with Convolutions (2014)
    """
    initializer = K.initializers.HeNormal()
    input_layer = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64,
                            padding='same',
                            activation='relu',
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            kernel_initializer=initializer)(input_layer)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(conv1)
    conv2 = K.layers.Conv2D(filters=192,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=initializer,
                            activation='relu')(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(conv2)
    filters = (64, 96, 128, 16, 32, 32)
    inception1 = inception_block(pool2, filters)
    filters = (128, 128, 192, 32, 96, 64)
    inception2 = inception_block(inception1, filters)
    pool3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(inception2)
    filters = (192, 96, 208, 16, 48, 64)
    inception3 = inception_block(pool3, filters)
    filters = (160, 112, 224, 24, 64, 64)
    inception4 = inception_block(inception3, filters)
    filters = (128, 128, 256, 24, 64, 64)
    inception5 = inception_block(inception4, filters)
    filters = (112, 144, 288, 32, 64, 64)
    inception6 = inception_block(inception5, filters)
    filters = (256, 160, 320, 32, 128, 128)
    inception7 = inception_block(inception6, filters)
    pool4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(inception7)
    filters = (256, 160, 320, 32, 128, 128)
    inception8 = inception_block(pool4, filters)
    filters = (384, 192, 384, 48, 128, 128)
    inception9 = inception_block(inception8, filters)
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=(1, 1),
                                         padding='valid')(inception9)
    dropout = K.layers.Dropout(rate=(0.4))(avg_pool)
    fc = K.layers.Dense(units=(1000), activation='softmax',
                        kernel_initializer=initializer)(dropout)

    model = K.Model(inputs=input_layer, outputs=fc)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
