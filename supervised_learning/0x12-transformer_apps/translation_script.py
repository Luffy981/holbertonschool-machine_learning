#!/usr/bin/env python3
"""translation script"""


import tensorflow_datasets as tfds
import tensorflow as tf
train_transformer = __import__('5-train').train_transformer
Dataset = __import__('3-dataset').Dataset
Translator = __import__('translator').Translator


tf.compat.v1.set_random_seed(0)
# parameters from paper (6, 512, 8, 2048, 40, 64, 20+)
transformer = train_transformer(4, 128, 8, 512, 40, 64, 20)
data = Dataset(32, 40)
translator = Translator(data, transformer)


test_set = tfds.load('ted_hrlr_translate/pt_to_en', split='test', as_supervised=True)

for pt, true_translation in test_set.take(5):
    translator.translate(pt.numpy().decode('utf-8'))
    print("Real translation: ", true_translation.numpy().decode('utf-8'), end="\n\n")
