#!/usr/bin/env python3
"""
Pooling Back Prop
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Args:
        dA: (m, h_new, w_new, c_new) containing the partial derivatives with
            respect to the output of the pooling layer
            m: is the number of examples
            h_new: is the height of the output
            w_new: is the width of the output
            c is: the number of channels
        A_prev: (m, h_prev, w_prev, c) containing the output of
                the previous layer
            h_prev: is the height of the previous layer
            w_prev: is the width of the previous layer
        kernel_shape: (kh, kw) containing the size of the kernel
                      for the pooling
            kh: is the kernel height
            kw: is the kernel width
        stride: is a tuple of (sh, sw) containing the strides for the pooling
            sh: is the stride for the height
            sw: is the stride for the width
        mode: is a string containing either max or avg
    Returns:
        the partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    der = np.zeros_like(A_prev)
    for frame in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    if mode == 'avg':
                        avg_dA = dA[frame, h, w, ch] / kh / kw
                        der[frame, sh*h:sh*h+kh,
                            sw*w:sw*w+kw, ch] += (np.ones((kh, kw)) * avg_dA)
                    if mode == 'max':
                        box = A_prev[frame, sh*h:sh*h+kh, sw*w:sw*w+kw, ch]
                        mask = (box == np.max(box))
                        der[frame, sh*h:sh*h+kh,
                            sw*w:sw*w+kw, ch] += (mask * dA[frame, h, w, ch])
    return der
