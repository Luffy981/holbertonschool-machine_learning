#!/usr/bin/env python3
"""
Function that creates a pd.DataFrame from a np.ndarray
"""


import pandas as pd


def from_numpy(array):
    """
    array is the np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame
    """
    pd.set_option('display.max_columns', None)
    df = pd.DataFrame(array)
    _, n = array.shape
    alphabet = range(65, 65 + n)
    alphabet = list(map(lambda x: chr(x), alphabet))
    df.columns = alphabet
    return df
