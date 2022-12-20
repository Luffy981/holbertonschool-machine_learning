#!/usr/bin/env python3
"""
Module contains function for performing
pca color augmentation on an image.
"""


import numpy as np


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the
    AlexNet paper.

    Args:
        image: tf.Tensor - Image to change.
        alphas: array of length 3 containing the amount that
        each channel should change.

    Return: tf.Tensor - Changed image.
    """

    image = image.numpy()
    img = image.reshape(-1, 3).astype(np.float32)
    scaling_factor = np.sqrt(3.0 / np.sum(np.var(img, axis=0)))
    img *= scaling_factor

    cov = np.cov(img, rowvar=False)
    U, S, V = np.linalg.svd(cov)

    rand = np.random.randn(3) * 0.1
    delta = np.dot(U, rand*S)
    delta = (delta * alphas * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

    img_out = np.clip(image + delta, 0, 255).astype(np.uint8)
    return img_out

if __name__ == "__main__":
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import save_img
    import numpy as np

    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(100)
    np.random.seed(100)
    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        save_img("./images/before_pca.jpg", image)
        alphas = np.random.normal(0, 0.1, 3)
        save_img("./images/pca.jpg", pca_color(image, alphas))
