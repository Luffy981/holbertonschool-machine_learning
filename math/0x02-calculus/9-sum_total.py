#!/usr/bin/env python3
"""SUMA POWER"""


def summation_i_squared(n):
    """sum squared"""
    if type(n) != int and n <= 0 and n is None:
        return None
    return (n * (n + 1) * (2 * n + 1)) / 6
