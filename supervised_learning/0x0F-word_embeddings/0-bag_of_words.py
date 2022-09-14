#!/usr/bin/env python3
"""Module contains function for creating bag of words."""


from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences: List of sentences to analyze.
        vocab: List of the vocabulary words to use
        for the analysis. If None, all words within
        sentences should be used.

    Return: embeddings, features
        embeddings: numpy.ndarray - (s, f) Embeddings.
            s: Number of sentences in sentences.
            f: Number of features analyzed.
        features: List of the features used for embeddings.
    """

    vec = CountVectorizer(vocabulary=vocab)
    X = vec.fit_transform(sentences)

    embeddings = X.toarray()
    features = vec.get_feature_names()

    return embeddings, features
