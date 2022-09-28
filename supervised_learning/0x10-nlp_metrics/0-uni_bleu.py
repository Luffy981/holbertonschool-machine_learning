#!/usr/bin/env python3
"""
Module contains function for calculating the unigram BLEU
score for a sentence generated by a model, compared to
reference sentences.
"""


import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence
    generated by a model.

    Args:
        references: List of reference translations, where each
        reference translation is a list of the words in the translation.
        sentence: List containing the model proposed sentence.

    Return: unigram BLEU score
    """

    tot, unigrams, bp = 0, len(sentence), 1
    sen = sentence.copy()

    min_ref = min([len(ref) for ref in references])

    if unigrams <= min_ref:
        bp = np.exp(1-min_ref/unigrams)

    while len(sen) > 0:
        wrd = sen[0]
        cnt = sen.count(wrd)
        for i in range(cnt):
            sen.pop(sen.index(wrd))

        mx_ref = max([ref.count(wrd) for ref in references])

        tot += cnt if cnt <= mx_ref else mx_ref

    return bp * (tot / unigrams)