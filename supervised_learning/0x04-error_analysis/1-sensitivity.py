#!/usr/bin/env python3
"""
Sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix
    Args:
        confusion: (classes, classes)-where row indices represent the
        correct labels and column indices represent the predicted labels
            classes:  is the number of classes
    Returns:
        (classes,)-containing the sensitivity of each class
    """
    sensitive = np.zeros((confusion.shape[0]))
    for x in range(confusion.shape[0]):
        # True positive
        # Total of sick individuals in population
        # print(confusion[x][x])
        # True positive + False negative
        # print(confusion[x])
        sensitive[x] = confusion[x][x] / np.sum(confusion[x])
    return sensitive
