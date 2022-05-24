#!/usr/bin/env python3
"""
Confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix
    Args:
        Labels: (m, classes)-containing the correct labels for each data point
            m: is the number of data points
            classes: is the number of classes
        Logits: (m, classes)-containing the predicted labels
    Returns:
    a confusion of shape (classes, classes) with row indices representing the
    correct labels and column indices representing the predicted labels
    """
    K = len(labels[0])
    # Initialize the confusion matrix
    result = np.zeros((K, K))
    # numpy.where(condition, [x, y, ]/)
    # Return elements chosen from x or y depending on condition.
    labelsn = np.where(labels == 1)[1]
    logitsn = np.where(logits == 1)[1]
    for i in range(len(labelsn)):
        result[labelsn[i]][logitsn[i]] += 1
    return result
