#!/usr/bin/env python3
"""Normal class"""


def factorial(number):
    """Method to calculate Factorial"""
    fact = 1
    for num in range(2, number + 1):
        fact = fact * num
    return fact


pi = 3.1415926536


def erf(t):
    """Error function"""
    sum = t - (t ** 3) / 3 + (t ** 5) / 10 - (t ** 7) / 42 + (t ** 9) / 216
    ERFC = (2 / (pi ** (0.5))) * sum
    return ERFC


class Normal:
    """Normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            self.mean = float(mean)
            self.stddev = float(stddev)
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            suma = 0
            for x in data:
                suma += (x - self.mean) ** 2
            self.stddev = (1 / (len(data)) * suma) ** 0.5

    def z_score(self, x):
        """Z-score given x-value"""
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """x-value of a given z-score"""
        x = (z * self.stddev) + self.mean
        return x

    def pdf(self, x):
        """Probability density function"""
        factor = 1 / (self.stddev * (2 * self.pi) ** 1/2)
        poew = (-1/2 * ((x - self.mean) / self.stddev) ** 2)
        pdf = factor * self.e ** poew
        return pdf

    def cdf(self, x):
        """Cumulative density function"""
        factor = (x - self.mean) / (self.stddev * (2 ** 0.5))
        CDF = (1 / 2) * (1 + erf(factor))
        return CDF
