#!/usr/bin/env python3
"""Accuracy of neural network"""


import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """cross-entropy loss using tf"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
