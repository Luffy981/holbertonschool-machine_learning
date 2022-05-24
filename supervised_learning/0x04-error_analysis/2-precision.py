#!/usr/bin/env python3
"""
Precision for confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    precision for each class in a confusion matrix
    Args:
        confusion: (classes, classes)-where row indices represent the
        correct labels and column indices represent the predicted labels
            classes: is the number of classes
    Returns:
        (classes,)-containing the precision of each class
    """
    precision = np.zeros((confusion.shape[0]))
    for x in range(confusion.shape[0]):
        # confusion[x][x] - true Positive
        # True positive + False positive = Column confusion
        precision[x] = confusion[x][x]/np.sum(confusion[:, x])
    return precision
