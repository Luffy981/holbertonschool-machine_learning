#!/usr/bin/env python3
"""
Pooling Forward Prop
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Args:
        A_prev: (m, h_prev, w_prev, c_prev) containing the output
                of the previous layer
            m: is the number of examples
            h_prev: is the height of the previous layer
            w_prev: is the width of the previous layer
            c_prev: is the number of channels in the previous layer
        kernel_shape:(kh, kw) containing the size of the kernel for the pooling
            kh: is the kernel height
            kw: is the kernel width
        stride: is a tuple of (sh, sw) containing the strides for the pooling
            sh: is the stride for the height
            sw: is the stride for the width
        mode: is a string containing either max or avg
    Returns:
        the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    if mode == 'max':
        op = np.max
    elif mode == 'avg':
        op = np.mean
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1
    pool = np.zeros((m, h_out, w_out, c_prev))
    for h in range(h_out):
        for w in range(w_out):
            pool[:, h, w, :] = op(A_prev[:, sh*h:sh*h+kh, sw*w:sw*w+kw],
                                  axis=(1, 2))
    return pool
