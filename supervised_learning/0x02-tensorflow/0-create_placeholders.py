#!/usr/bin/env python3
"""Placeholders"""


import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """Function create two placeholders, x and y, for the neural network
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
    """
    # x is the placeholder for the input data to the neural network
    x = tf.placeholder(name="x", shape=(None, nx), dtype=tf.float32)
    # y is the placeholder for the one-hot labels for the input data
    y = tf.placeholder(name="y", shape=(None, classes), dtype=tf.float32)
    return x, y
