#!/usr/bin/env python3
"""
Learning rate decay
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
       Updates the learning rate using inverse time decay in numpy.
       Args:
         alpha: original learning rate
         decay_rate: weight used to determine the rate at which
           alpha will decay
         global_step: number of passes of gradient descent that have elapsed
         decay_step: number of passes of gradient descent that should occur
           before alpha is decayed further
       Returns:
         The updated value for alpha.
    """
    learning_rate = (1+(decay_rate*int(global_step/decay_step)))
    return alpha/learning_rate
