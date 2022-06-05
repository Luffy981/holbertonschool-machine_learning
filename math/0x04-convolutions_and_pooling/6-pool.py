#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Args:
        images: (m,h,w) containing multiple grayscale images
            m: is the number of images
            h: is the height in pixels of the images
            w: is the width in pixels of the images
            c: is the number of channels in the image
        kernel: (kh,kw) containing the kernel for the convolution
            kh: is the height of the kernel
            kw: is the width of the kernel
    Return:
        a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh, kw = kernel_shape
    sh, sw = stride
    if mode == "max":
        op = np.amax
    else:
        op = np.average
    # Calculatin output shape
    p_h, p_w = 0, 0
    W_out = (w - kw + (2 * p_w)) // sw + 1
    H_out = (h - kh + (2 * p_h)) // sh + 1
    output_matriz = np.zeros((m, H_out, W_out, c))
    for h in range(H_out):
        for w in range(W_out):
            output_matriz[:, h, w, :] = op(
                images[:, sh*h: sh*h+kh, sw*w:sw*w+kw, :], axis=(1, 2)
                )
    return output_matriz
