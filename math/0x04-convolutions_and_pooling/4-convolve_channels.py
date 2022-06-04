#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
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
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh, sw = stride
    if padding == 'same':
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    new_padded_images = np.pad(images, ((0, 0),
                                        (ph, ph),
                                        (pw, pw),
                                        (0, 0)), 'constant')
    o_h = ((h + (2 * ph) - kh) // sh) + 1
    o_w = ((w + (2 * pw) - kw) // sw) + 1
    output = np.zeros((m, o_h, o_w))
    for x in range(o_w):
        for y in range(o_h):
            i = y * sh
            j = x * sw
            mat = new_padded_images[:, i:i+kh, j:j+kw, :]
            output[:, y, x] = np.sum(np.multiply(mat, kernel), axis=(1, 2, 3))
    return output
