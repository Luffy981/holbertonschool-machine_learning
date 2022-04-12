#!/usr/bin/env python3
def add_arrays(arr1, arr2):
    sum = []
    try:
        for i in range(len(arr1)):
            sum.append(arr1[i] + arr2[i])
    except Exception:
        return None
    return sum
