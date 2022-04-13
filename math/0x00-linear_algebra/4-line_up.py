#!/usr/bin/env python3
"""function add"""


def add_arrays(arr1, arr2):
    """
    add matrix
    """
    sum = []
    try:
        for i in range(len(arr1)):
            sum.append(arr1[i] + arr2[i])
    except Exception:
        return None
    return sum
