#!/usr/bin/env python3
"""
Creates the class SelfAttention
"""


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Class to encode for machine translation
    """
    def __init__(self, units):
        """
        Class constructor
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        s_prev is a tensor of shape (batch, units) containing the previous
        decoder hidden state
        hidden_states is a tensor of shape (batch, input_seq_len, units)
        containing the outputs of the encoder
        Returns: context, weights
            context is a tensor of shape (batch, units) that contains
            the context vector for the decoder
            weights is a tensor of shape (batch, input_seq_len, 1) that
            contains the attention weights
        """
        W = self.W(tf.expand_dims(s_prev, 1))
        U = self.U(hidden_states)
        V = self.V(tf.nn.tanh(W + U))
        weights = tf.nn.softmax(V, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
