#!/usr/bin/env python3
"""
Script to alter the pd.DataFrame such that the rows and columns are
transposed and the data is sorted in reverse chronological order
"""


import pandas as pd
from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.sort_values(by=['Timestamp'], ascending=False).transpose()

print(df.tail(10))
