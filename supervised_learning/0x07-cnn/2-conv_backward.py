#!/usr/bin/env python3
"""
Convolutional Back Prop
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Args:
        dZ: (m, h_new, w_new, c_new) containing the partial derivatives with
            respect to the unactivated output of the convolutional layer
            m: is the number of examples
            h_new: is the height of the output
            w_new: is the width of the output
            c_new: is the number of channels in the output
        A_prev: (m, h_prev, w_prev, c_prev) containing the output of the
                previous layer
            h_prev: is the height of the previous layer
            w_prev: is the width of the previous layer
            c_prev: is the number of channels in the previous layer
        W: (kh, kw, c_prev, c_new) containing the kernels for the convolution
            kh: is the filter height
            kw: is the filter width
        b: (1, 1, 1, c_new) containing the biases applied to the convolution
        padding: is a string that is either same or valid
        stride: (sh, sw) containing the strides for the convolution
            sh: is the stride for the height
            sw: is the stride for the width
    Returns:
        the partial derivatives with respect to the previous layer (dA_prev),
        the kernels (dW), and the biases (db), respectively
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    m, h_new, w_new, c_new = dZ.shape
    dW = np.zeros_like(W)
    da = np.zeros_like(A_prev)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    if padding == 'valid':
        p_h, p_w = 0, 0
    elif padding == 'same':
        p_h = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        p_w = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1
    A_prev = np.pad(A_prev, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)),
                    mode='constant', constant_values=0)
    dA = np.pad(da, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)),
                mode='constant', constant_values=0)
    for frame in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c_new):
                    kernel = W[:, :, :, k]
                    dz = dZ[frame, h, w, k]
                    mat = A_prev[frame, sh*h:sh*h+kh, sw*w:sw*w+kw, :]
                    dW[:, :, :, k] += mat * dz
                    dA[frame, sh*h:sh*h+kh,
                       sw*w:sw*w+kw, :] += np.multiply(kernel, dz)
    if padding == 'same':
        dA = dA[:, p_h: -p_h, p_w: -p_w, :]
    return dA, dW, db
