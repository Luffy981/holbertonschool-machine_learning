#!/usr/bin/env python3
"""
Module contains function for rotating
an image 90 degrees counter-clockwise.
"""


import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise.

    Args:
        image: tf.Tensor - Image to rotate.

    Return: tf.Tensor - rotated image.
    """
    return tf.image.rot90(image)


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import save_img

    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(2)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        save_img("./images/before_rotation.jpg", image)
        save_img("./images/rotation.jpg", rotate_image(image))
