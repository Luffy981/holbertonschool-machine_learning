#!/usr/bin/env python3
"""
Module contains function that converts a gensim word2vec
model to a keras Embedding layer.
"""


import tensorflow.keras


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras
    embedding layer.

    Args:
        model: Trained gensim word2vec model.

    Return: Trained keras embedding.
    """
    embedding = model.wv.get_keras_embedding(train_embeddings=True)
    return embedding
