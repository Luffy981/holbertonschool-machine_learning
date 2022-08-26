#!/usr/bin/env python3
"""
Module contains funtion for creating convolutional
autoencoder.
"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates convolutional autoencoder.

    Args:
        input_dims: Tuple of integers - imensions of the model input.
        filters: List - number of filters for each convolutional layer
        in the encoder.
        latent_dims: Tuple of integers - dimensions of the latent space
        representation.

    Return: encoder, decoder, autoencoder
        encoder: Encoder model.
        decoder: Decoder model.
        autoencoder: The full autoencoder model.
    """
    Conv2d = keras.layers.Conv2D
    MaxPool = keras.layers.MaxPool2D
    UpSample = keras.layers.UpSampling2D

    encoder_input = keras.Input(shape=input_dims)
    decoder_input = keras.Input(shape=latent_dims)

    encoded_layer = Conv2d(
        filters[0], (3, 3), activation="relu", padding="same")(encoder_input)
    encoded_layer = MaxPool((2, 2), padding="same")(encoded_layer)

    for filter in filters[1:]:
        encoded_layer = Conv2d(
            filter, (3, 3), padding="same", activation="relu")(encoded_layer)
        encoded_layer = MaxPool((2, 2), padding="same")(encoded_layer)

    decoded_layer = Conv2d(
        filters[-1], (3, 3), padding="same", activation="relu")(decoder_input)
    decoded_layer = UpSample((2, 2))(decoded_layer)

    for x, filter in enumerate(list(reversed(filters))[1:]):
        if x == len(filters) - 2:
            decoded_layer = Conv2d(
                filter, (3, 3), padding="valid",
                activation="relu")(decoded_layer)
        else:
            decoded_layer = Conv2d(
                filter, (3, 3), padding="same",
                activation="relu")(decoded_layer)

        decoded_layer = UpSample((2, 2))(decoded_layer)

    decoded_layer = Conv2d(
        input_dims[-1], (3, 3), padding="same", activation="sigmoid"
        )(decoded_layer)

    encoder = keras.Model(encoder_input, encoded_layer)

    decoder = keras.Model(decoder_input, decoded_layer)

    autoencoder = keras.Model(encoder_input, decoder(encoder(encoder_input)))

    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, autoencoder
