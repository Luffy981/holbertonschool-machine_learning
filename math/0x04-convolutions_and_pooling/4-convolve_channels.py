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
    c = images.shape[3]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    kc = kernel.shape[2]
    sh, sw = stride
    # Calculatin output shape
    if padding == 'same':
        p_h = ((((h - 1) * sh) + kh - h) // 2) + 1
        p_w = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        p_h = 0
        p_w = 0
    elif type(padding) == tuple:
        p_h, p_w = padding
    W_out = ((w + (2 * p_w) - kw) // sw) + 1
    H_out = ((h + (2 * p_h) - kh) // sh) + 1
    p_images = np.pad(images, ((0, 0),
                               (p_h, p_h),
                               (p_w, p_w),
                               (0, 0)),
                      'constant')
    output_matriz = np.zeros((m, H_out, W_out))
    for w in range(W_out):
        for h in range(H_out):
            # np.tensordot(a2D,a3D,((-1,),(-1,))).transpose(1,0,2)
            part_image = p_images[:, sh*h:sh*h + kh, sw*w:sw*w + kw, :]
            output_matriz[:, h, w] = np.tensordot(part_image, kernel, axes=3)
    return output_matriz
