#!/usr/bin/env python3
"""
Translator class for machine translation
using a transformer.
"""


import tensorflow as tf
create_masks = __import__('4-create_masks').create_masks


class Translator():

    def __init__(self, data, transformer):
        self.data = data
        self.transformer = transformer

    def evaluate(self, inp_sentence):
        start_token = [self.data.tokenizer_pt.vocab_size]
        end_token = [self.data.tokenizer_pt.vocab_size + 1]

        inp_sentence = start_token + self.data.tokenizer_pt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        decoder_input = [self.data.tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)
        
        for i in range(self.data.max_len):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)
    
            predictions, attention_weights = \
            self.transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
            
            predictions = predictions[: ,-1:, :]

            pred_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
            if pred_id == self.data.tokenizer_en.vocab_size+1:
                return tf.squeeze(output, axis=0), attention_weights
            
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, pred_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(self, sentence):
        result, attention_weights = self.evaluate(sentence)

        predicted_sentence = \
        self.data.tokenizer_en.decode([i for i in result if i < self.data.tokenizer_en.vocab_size])

        print('Input: {}'.format(sentence))
        print('Prediction: {}'.format(predicted_sentence))