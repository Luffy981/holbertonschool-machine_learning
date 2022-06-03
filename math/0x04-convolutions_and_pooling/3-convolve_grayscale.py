
  
#!/usr/bin/env python3
"""
   Module contains
   convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
       Performs a valid convolution on grayscale images.
       Args:
         images: numpy.ndarray - (m, h, w) contains multiple greyscale images
         kernel: numpy.ndarray - (kh, kw) kernel for convolution
         padding: (ph, pw) tuple of padding dimensions
         stride: (sh, sw) tuple of stride dimensions
       Returns:
         numpy.ndarray - convolved images
    """
    samples, samp_h, samp_w = images.shape
    filter_h, filter_w = kernel.shape
    sh, sw = stride

    if type(padding) is tuple:
        pad_h, pad_w = padding
    elif padding == "valid":
        pad_h, pad_w = 0, 0
    elif padding == "same":
        pad_h = (((samp_h - 1) * sh) + filter_h - samp_h) // 2 + 1
        pad_w = (((samp_w - 1) * sw) + filter_w - samp_w) // 2 + 1

    padded = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant'
        )

    out_h = (samp_h + (2 * pad_h) - filter_h) // sh + 1
    out_w = (samp_w + (2 * pad_w) - filter_w) // sw + 1

    conv = np.zeros((samples, out_h, out_w))

    for h in range(out_h):
        for w in range(out_w):
            conv[:, h, w] = (
                kernel * padded[:, sh*h: sh*h+filter_h, sw*w:sw*w+filter_w]
                ).sum(axis=(1, 2))
    return conv#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
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
    sh, sw = stride
    # Calculatin output shape
    if padding == 'same':
        W_out = w
        H_out = h
        ph = ((((h - 1) * sh) + kh - h) // 2) + 1
        pw = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding == 'valid':
        p_h = 0
        p_w = 0
    W_out = (w - kw + (2 * p_w)) // sw + 1
    H_out = (h - kh + (2 * p_h)) // sh + 1
    p_images = np.pad(images, ((0, 0), (p_h, p_h), (p_w, p_w)), 'constant')
    output_matriz = np.zeros((m, H_out, W_out))
    for i in range(W_out):
        for j in range(H_out):
            # np.tensordot(a2D,a3D,((-1,),(-1,))).transpose(1,0,2)
            # o[:,j,i]= (kernel * images[:,j:j+kh, i:i+kw]).sum(axis=(1,2))
            part_image = p_images[:, sh*j:sh*j + kh, sw*i:sw*i + kw]
            output_matriz[:, j, i] = np.tensordot(part_image, kernel, axes=2)
    return output_matriz
