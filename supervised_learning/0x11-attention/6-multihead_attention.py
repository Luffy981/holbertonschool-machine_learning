#!/usr/bin/env python3
"""
Function that calculates the scaled dot product attention
res:
https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
"""


import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class to perform multi head attention
    """

    def __init__(self, dm, h):
        """
        Class constructor
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Q is a tensor of shape (batch, seq_len_q, dk) containing the input
        to generate the query matrix
        K is a tensor of shape (batch, seq_len_v, dk) containing the input
        to generate the key matrix
        V is a tensor of shape (batch, seq_len_v, dv) containing the input
        to generate the value matrix
        mask is always None
        Returns: output, weights
            outputa tensor with its last two dimensions as
            (..., seq_len_q, dm) containing the scaled dot product attention
            weights a tensor with its last three dimensions as
            (..., h, seq_len_q, seq_len_v) containing the attention weights
        """
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1,
                                                                3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))
        output = self.linear(concat_attention)
        return output, attention_weights
