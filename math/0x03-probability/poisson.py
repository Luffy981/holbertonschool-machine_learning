#!/usr/bin/env python3
"""class poisson"""


class Poisson:
    """Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Constructor to poisson"""
        self.lambtha = float(lambtha)
        self.e = 2.7182818285
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            sum_P = 0
            prob = []
            for x in data:
                factor = (self.lambtha ** x)
                sum_P += (self.e**(-self.lambtha) * factor) / factorial(x)
                prob.append(sum_P)
            self.lambtha = round(sum_P, 2)

    def factorial(number):
        """Method to calculate Factorial"""
        fact = 1
        for num in range(2, number + 1):
            fact = fact * num
        return fact

    def pmf(self, k):
        """probability mass function"""
        try:
            k = int(k)
        except Exception:
            return 0
        PMF = (self.e**(-self.lambtha) * (self.lambtha ** k)) / factorial(k)
        return PMF
