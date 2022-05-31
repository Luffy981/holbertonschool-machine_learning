#!/usr/bin/env python3
"""Save and Load Model"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Args:
        network: is the model to save
        filename: is the path of the file that the model should be saved to
    Returns:
        None
    """
    K.models.save_model(model=network, filepath=filename)
    return None


def load_model(filename):
    """
    Args:
        filename: is the path of the file that the model should be loaded from
    Returns:
        the loaded model
    """
    return K.models.load_model(filepath=filename)
