#!/usr/bin/env python3
"""
specificity for each class in confusion matrix
"""
import numpy as np

def specificity(confusion):
    """
    specificity for each class in a confusion matrix
    Args:
        confusion: (classes, classes)-where row indices represent the
        correct labels and column indices represent the predicted labels
            classes: is the number of classes
    Returns:
        (classes,)-containing the precision of each class
    """
    # ndarray.flatten(order='C')
    # Return a copy of the array collapsed into one dimension.
    cSum = np.sum(confusion.flatten())
    # scpecificity = true negatives / true negatives + false positives
    specificity = np.zeros((confusion.shape[0]))
    for x in range(confusion.shape[0]): 
        true_negative = cSum - (np.sum(confusion[x]) + np.sum(confusion[:, x])
                                 - confusion[x][x])
        specificity[x] = true_negative / (true_negative +
         (np.sum(confusion[:, x]) - confusion[x][x]))
    
    return specificity

