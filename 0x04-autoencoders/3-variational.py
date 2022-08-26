#!/usr/bin/env python3
"""
Module contains function for creating variational autoencoder.
"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates variational autoencoder network.

    Args:
        input_dims: Integer containing the dimensions of the model input.
        hidden_layers: List containing the number of nodes for each hidden
        layer in the encoder, respectively.
        latent_dims: Integer containing the dimensions of the latent space
        representation.

    Return: encoder, decoder, autoencoder
        encoder: Encoder model, which should output the latent representation,
        the mean, and the log variance, respectively.
        decoder: Decoder model.
        autoencoder: Full autoencoder model.
    """
    K = keras.backend

    def sampling(args):
        """Samples similar points from latent space."""
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], latent_dims), mean=0, stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    Dense = keras.layers.Dense
    Loss = keras.losses.binary_crossentropy

    input_layer = keras.Input(shape=(input_dims,))
    encoded_input = keras.Input(shape=(latent_dims,))
    decoded_input = encoded_input

    encoded_layers = Dense(hidden_layers[0], activation="relu")(input_layer)

    for nodes in hidden_layers[1:]:
        encoded_layers = Dense(nodes, activation="relu")(encoded_layers)

    encoded_layers = Dense(latent_dims)(encoded_layers)
    z_mean = Dense(latent_dims)(encoded_layers)
    z_log_sigma = Dense(latent_dims)(encoded_layers)
    Z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    decoded = Dense(hidden_layers[-1], activation="relu")(decoded_input)

    for x, nodes in enumerate(list(reversed(hidden_layers[:-1]))+[input_dims]):
        if x == len(hidden_layers) - 1:
            decoded = Dense(nodes, activation="sigmoid")(decoded)
        else:
            decoded = Dense(nodes, activation="relu")(decoded)

    encoder = keras.Model(input_layer, [z_mean, z_log_sigma, Z])
    decoder = keras.Model(encoded_input, decoded)
    outputs = decoder(encoder(input_layer)[2])
    autoencoder = keras.Model(input_layer, outputs)

    reconstruction_loss = Loss(input_layer, outputs)
    reconstruction_loss *= input_dims
    k1_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    k1_loss = K.sum(k1_loss, axis=-1)
    k1_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + k1_loss)
    autoencoder.add_loss(vae_loss)
    autoencoder.compile(optimizer="adam")

    return encoder, decoder, autoencoder
