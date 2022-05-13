#!/usr/bin/env python3
"""Accuracy of neural network"""


import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """cross-entropy loss using tf"""
    cce = tf.keras.losses.CategoricalCrossentropy(
        name='softmax_cross_entropy_loss')
    return cce(y, y_pred)
