#!/usr/bin/env python3
"""
Model To JSON
Load from JSON
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    save model to JSON file
    """
    model = network.to_json()
    with open(filename, 'w') as f:
        f.write(model)
    return None


def load_config(filename):
    """
    Load from JSON file
    """
    with open(filename, 'r') as f:
        model = f.read()
    return K.models.model_from_json(model)
