#!/usr/bin/env python3
"""Save and Load Weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Args:
        network: is the model whose weights should be saved
        filename: is the path of the file that the weights should be saved to
        save_format: is the format in which the weights should be saved
    Returns:
        None
    """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Args:
        network: is the model to which the weights should be loaded
        filename: is the path of the file that the weights should be loaded
    Returns:
        None
    """
    return network.load_weights(filepath=filename)
