#!/usr/bin/env python3
"""Module contains function for creating TF-IDF embedding."""


from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding.

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

    vec = TfidfVectorizer(use_idf=True, vocabulary=vocab)
    X = vec.fit_transform(sentences)

    embeddings = X.toarray()
    features = vec.get_feature_names()

    return embeddings, features
