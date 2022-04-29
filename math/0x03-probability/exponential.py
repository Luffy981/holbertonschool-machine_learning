#!/usr/bin/env python3
"""Exponential class"""


class Exponential:
    """Exponential Distribution"""
    def __init__(self, data=None, lambtha=1.):
        self.e = 2.7182818285
        if data is None:
            self.lambtha = float(lambtha)
            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            count = 0
            for x in data:
                count += x
            self.lambtha = 1 / float(count / len(data))

    def pdf(self, k):
        """probability mass function"""
        try:
            k = int(k)
        except Exception:
            return 0
        PDF = self.lambtha * self.e ** (-self.lambtha * k)
        return PDF

    def cdf(self, x):
        """Cumulative density function"""
        try:
            x = int(x)
        except Exception:
            return 0
        CDF = 1 - self.e ** (-self.lambtha * x)
        return CDF
