#!/usr/bin/env python3
"""
Module contains function for randomly
changing brightness of an image.
"""


import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes brightness of an image.

    Args:
        image: tf.Tensor - Image to change.
        max_delta: maximum amount the image should be
        brightened (or darkened).

    Return: tf.Tensor - Changed image.
    """
    return tf.image.random_brightness(image, max_delta)


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import save_img

    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(4)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        save_img("./images/before_brightness.jpg", image)
        save_img("./images/brightness.jpg", change_brightness(image, 0.3))
