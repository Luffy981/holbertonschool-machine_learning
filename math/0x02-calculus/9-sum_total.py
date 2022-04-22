#!/usr/bin/env python3
"""SUMA POWER"""


def summation_i_squared(n):
    """ Returns the summation of the given number of squared"""
    if n is None or n <= 0:
        return None
    return n * (n + 1) * (2 * n + 1) / 6
