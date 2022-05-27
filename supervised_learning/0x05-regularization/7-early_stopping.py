#!/usr/bin/env python3
"""
Early Stopping
"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
       Determines stop gradient descent early.
       Args:
         cost: Current validation cost
         opt_cost: Lowest recorded validation cost
         threshold: Threshold used for early stopping
         patience: Patience count used for early stopping
         count: How long threshold has not been met
       Returns:
         Boolean on whether network should be updated
           followed by update count.
    """
    cost_step = opt_cost - cost
    if cost_step > threshold:
        count = 0
    else:
        count += 1
    if count < patience:
        return False, count
    else:
        return True, count
