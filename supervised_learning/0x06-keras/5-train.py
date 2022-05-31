#!/usr/bin/env python3
"""
Train deep neural network with  Keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Training model
    Args:
        network: is the model to train
        data: (m, nx) containing the input data
        labels:(m, classes) containing the labels of data
        batch_size: is the size of the batch used for mini-batch
                  gradient descent
        epochs: is the number of passes through data for mini-batch
                gradient descent
        verbose: is a boolean that determines if output should be
                 printed during training
        shuffle: is a boolean that determines whether to shuffle
                 the batches every epoch.
        validation_data: the data to validate the model with, if not None
    Returns:
        the History object generated after training the model
    """

    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data)
