#!/usr/bin/env python3
"""
Module contains function that changes
the hue of an image.
"""


import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: tf.Tensor - Image to change.
        max_delta: amount the hue should change.

    Return: tf.Tensor - Changed image.
    """
    return tf.image.adjust_hue(image, delta)


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import save_img

    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(5)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        save_img("./images/before_hue.jpg", image)
        save_img("./images/hue.jpg", change_hue(image, -0.5))
