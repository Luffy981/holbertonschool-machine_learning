#/usr/bin/env python3
"""
Creates the class RNNEncoder
"""


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Class to encode for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                recurrent_initializer="glorot_uniform",
                return_sequences=True,
                return_state=True)

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros
        Returns: a tensor of shape (batch, units) containing the initialized
        hidden states
        """
        initializer = tf.keras.initializers.Zeros()
        values = initializer(shape=(self.batch, self.units))
        return values

    def call(self, x, initial):
        """
        x is a tensor of shape (batch, input_seq_len) containing the input to
        the encoder layer as word indices within the vocabulary
        initial is a tensor of shape (batch, units) containing the initial
        hidden state
        Returns: outputs, hidden
            outputs is a tensor of shape (batch, input_seq_len,
                    units)containing the outputs of the encoder
            hidden is a tensor of shape (batch, units) containing
            the last hidden state of the encoder
        """
        inputs = self.embedding(x)
        outputs, hidden = self.gru(inputs, initial_state=initial)
        return outputs, hidden
