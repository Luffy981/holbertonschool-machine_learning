#!/usr/bin/env python3
"""Exponential class"""


class Exponential:
    """Exponential Distribution"""
    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)
        self.e = 2.7182818285
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            for x in data:
                self.lambtha += self.lambtha * self.e ** (-self.lambtha * x)
