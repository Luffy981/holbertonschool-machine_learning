#!/usr/bin/env python3
"""
    Module contains
    f1_score(confusion):
"""


import numpy as np


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
        Calculates the F1 score of a confusion matrix.
        Args:
          confusion: numpy.ndarray (classes, classes) - where row
            indices represent the correct labels and column indices
            represent the predicted labels.
        Returns:
          numpy.ndarray (classes,) - containing the f1 score
            of each class.
    """
    p, s = precision(confusion), sensitivity(confusion)
    return 2*((p*s)/(p+s))
