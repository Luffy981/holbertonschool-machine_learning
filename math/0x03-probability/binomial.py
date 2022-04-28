#!/usr/bin/env python3
"""Binomial class"""


class Binomial:
    """Binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        self.p = float(p)
        self.n = float(n)
        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

    def factorial(number):
        """Method to calculate Factorial"""
        fact = 1
        for num in range(2, number + 1):
            fact = fact * num
        return fact
