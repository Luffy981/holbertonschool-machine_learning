#!/usr/bin/env python3
"""Only just one neuron"""


import numpy as np


class Neuron:
    """Neuron logic"""
    def __init__(self, nx):
        """Initialize neuron"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # np.random.normal(loc=mean,scale=standard_deviation, size=samples)
        # Loc: This parameter defaults to 0
        # Scale: By default, the scale parameter is set to 1.
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
