#!/usr/bin/env python3
import numpy as np
"""performs a valid convolution on grayscale images"""


def convolve_grayscale_valid(images, kernel):
    """
    Args:
        images: (m,h,w) containing multiple grayscale images
            m: is the number of images
            h: is the height in pixels of the images
            w: is the width in pixels of the images
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
    # Calculatin output shape
    W_out = w - kw + 1
    H_out = h - kh + 1
    output_matriz = np.zeros((m, H_out, W_out))
    for i in range(W_out):
        for j in range(H_out):
            # np.tensordot(a2D,a3D,((-1,),(-1,))).transpose(1,0,2)
            part_image = images[:, j:j + kh, i:i + kw].T
            output_matriz[:, j, i] = np.tensordot(kernel,
                                                  part_image,
                                                  axes=2)
    return output_matriz