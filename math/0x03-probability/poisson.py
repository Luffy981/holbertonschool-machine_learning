#!/usr/bin/env python3
"""class poisson"""


def factorial(number):
    """Method to calculate Factorial"""
    fact = 1
    for num in range(2, number + 1):
        fact = fact * num
    return fact


class Poisson:
    """Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Constructor to poisson"""
        self.e = 2.7182818285
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            count = 0
            prob = []
            for x in data:
                count += x
            self.lambtha = float(count / len(data))

    def pmf(self, k):
        """probability mass function"""
        try:
            k = int(k)
        except Exception:
            return 0
        PMF = (self.e**(-self.lambtha) * (self.lambtha ** k)) / factorial(k)
        return PMF

    def cdf(self, k):
        """Cumulative density function"""
        try:
            k = int(k)
        except Exception:
            return 0
        CDF = 0
        for i in range(k + 1):
            factor = (self.e ** (-self.lambtha) * (self.lambtha ** i))
            CDF += factor / factorial(i)
        return CDF
