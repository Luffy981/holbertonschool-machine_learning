#!/usr/bin/env python3
"""
Function that loads data from a file as a pd.DataFrame
"""


import pandas as pd


def from_file(filename, delimiter):
    """
    Loads from a file as a dataframe
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
