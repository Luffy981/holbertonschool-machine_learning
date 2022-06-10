#!/usr/bin/env python3
"""
Forward propagation in CNN
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Args:
        A_prev: (m, h_prev, w_prev, c_prev) containing the output of the
                previous layer
            m: is the number of examples
            h_prev: is the height of the previous layer
            w_prev: is the width of the previous layer
            c_prev: is the number of channels in the previous layer
        W: (kh, kw, c_prev, c_new) containing the kernels for the convolution
            kh: is the filter height
            kw: is the filter width
            c_prev: is the number of channels in the previous layer
            c_new: is the number of channels in the output
        b: (1, 1, 1, c_new) containing the biases applied to the convolution
        activation: is an activation function applied to the convolution
        padding: is a string that is either same or valid,
                 indicating the type of padding used
        stride: (sh, sw) containing the strides for the convolution
            sh: is the stride for the height
            sw: is the stride for the width
    Return:
        the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        p_h = ((((h_prev - 1) * sh) + kh - h_prev) // 2)
        p_w = ((((w_prev - 1) * sw) + kw - w_prev) // 2)
    elif padding == 'valid':
        p_h, p_w = 0, 0
    elif type(padding) == tuple:
        p_h, p_w = padding
    p_images = np.pad(A_prev, ((0, 0),
                               (p_h, p_h),
                               (p_w, p_w),
                               (0, 0)),
                      'constant')
    h_out = ((h_prev - kh + 2 * p_h) // sh) + 1
    w_out = ((w_prev - kw + 2 * p_w) // sw) + 1
    conv = np.zeros((m, h_out, w_out, c_new))
    for k in range(c_new):
        for h in range(h_out):
            for w in range(w_out):
                part_image = p_images[:, sh*h:sh*h+kh, sw*w:sw*w+kw]
                conv[:, h, w, k] = np.tensordot(part_image,
                                                W[:, :, :, k], axes=3) + b[:,
                                                                           :,
                                                                           :,
                                                                           k]
    return activation(conv)
