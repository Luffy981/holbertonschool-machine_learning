#!/usr/bin/env python3
"""Derivates"""


def poly_derivative(poly):
    """Derivates"""
    derivate = []
    for i in range(len(poly)):
        if i >= 1:
            derivate.append(poly[i] * i)
    return derivate
