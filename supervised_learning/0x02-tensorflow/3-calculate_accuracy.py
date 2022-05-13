#!/usr/bin/env python3
"""Accuracy of neural network"""


import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Calculate accuracy of neural network
    y: is a placeholder for the labels of the input data
    y_pred: is a tensor containing the network’s predictions
    accuracy = correct_predictions / all_predictions
    """
    # argmax: Returns the index with the largest value across axes of a tensor
    real_prediction = tf.math.argmax(y_pred, axis=1)
    neural_prediction = tf.math.argmax(y, axis=1)
    # Returns the truth value of (x == y) element-wise.
    equality = tf.math.equal(real_prediction, neural_prediction)
    # Computes the mean of elements across dimensions of a tensor.
    # Casts a tensor to a new type.
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
