#!/usr/bin/env python3
"""
Function that calculates the positional encoding for a transformer
"""


import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    max_seq_len is an integer representing the maximun sequence length
    dm is the model depth
    returns:
        A numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors
    """
    pos_encoding = np.zeros([max_seq_len, dm])
    for i in range(dm):
        for pos in range(max_seq_len):
            pos_encoding[pos, i] = pos / np.power(10000, (2 * (i // 2) / dm))
    # dimension 2i
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    # dimension 2i + 1
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
    return pos_encoding
