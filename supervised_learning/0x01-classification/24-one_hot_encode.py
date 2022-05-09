#!/usr/bin/env python3
"""One hot encoding"""


from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix"""
    data = np.array(Y).reshape(classes, 1)
    encoder = OneHotEncoder(sparse=False)
    one_hot = encoder.fit_transform(data)
    return one_hot
