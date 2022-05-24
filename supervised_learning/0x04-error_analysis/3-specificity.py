#!/usr/bin/env python3
"""
    Module contains
    specificity(confusion):
"""


import numpy as np


def specificity(confusion):
    """
        Calculates the specificity for each class in
          a confusion matrix
        Args:
          confusion: numpy.ndarray (classes, classes) - where row
            indices represent the correct labels and column indices
            represent the predicted labels.
        Returns:
          numpy.ndarray (classes,) - containing the specificity
            of each class.
    """
    cSum = np.sum(confusion.flatten())
    specificity = np.zeros((confusion.shape[0]))
    for x in range(confusion.shape[0]):
        t_neg = cSum - (
            np.sum(confusion[x]) + np.sum(confusion[:, x]) -
            confusion[x][x]
            )
        specificity[x] = t_neg/(
          np.sum(confusion[:, x]) - confusion[x][x] + t_neg
          )
    return specificity
