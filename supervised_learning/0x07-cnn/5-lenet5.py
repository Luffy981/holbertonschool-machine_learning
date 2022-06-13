#!/usr/bin/env python3
"""
LeNet-5 (Tensorflow 1)
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Args:
        x: is a tf.placeholder of shape (m, 28, 28, 1)
           containing the input images for the network
            m: is the number of images
    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    initializer = K.initializers.HeNormal()
    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=(5, 5),
                            padding='same',
                            activation='relu',
                            kernel_initializer=initializer)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=(5, 5),
                            padding='valid',
                            activation='relu',
                            kernel_initializer=initializer)(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv2)
    flat = K.layers.Flatten()(pool2)
    fc1 = K.layers.Dense(units=120,
                         kernel_initializer=initializer,
                         activation='relu')(flat)
    fc2 = K.layers.Dense(units=84,
                         kernel_initializer=initializer,
                         activation='relu')(fc1)
    fc3 = K.layers.Dense(units=10,
                         kernel_initializer=initializer,
                         activation='softmax')(fc2)
    model = K.Model(inputs=X, outputs=fc3)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
