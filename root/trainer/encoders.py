# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import unicodedata
import string


class Encoder(object):

    def __init__(self, alphabet, maxlen, num_classes, clear_accents=True):
        self.alphabet = alphabet
        self.vocabulary = set(self.alphabet)
        self.vocab_size = len(self.vocabulary)
        # reserve 0 for spaces and unknown characters, hence index + 1:
        self.mapping = {char: (index + 1) for index, char in enumerate(sorted(self.vocabulary))}
        self.reverse_mapping = {(index + 1): char for index,
                                char in enumerate(sorted(self.vocabulary))}
        # self.label_encoder = LabelEncoder()
        self.maxlen = maxlen
        self.num_classes = num_classes
        self.clear_accents = clear_accents
        # white list german umlauts in alphabet
        self.umlaut_white_list = set([umlaut for umlaut in 'ÄÖÜäöü' if umlaut in self.vocabulary])
        self.lowercase = self.vocabulary.isdisjoint(set(string.ascii_uppercase + 'ÄÖÜ'))

    def remove_accents(self, s):
        """
        Remove accents from unicode strings, thanks to:
        http://stackoverflow.com/a/518232/2809427
        """
        # normalize text and split umlauts, accents, etc. from characters
        s_normalized = ''.join(
            c if c in self.umlaut_white_list else unicodedata.normalize('NFD', c) for c in s)
        # discard detached accents
        return ''.join(c for c in s_normalized if unicodedata.category(c) != 'Mn')

    def encode_data(self, x, backwards=False):
        """Encode texts into one-hot vectors:"""

        # numerical encoding
        encoded_items = []
        for item in x:
            # check if any caps in vacobulary
            if self.lowercase:
                item = item.lower()
            if self.clear_accents:
                item = self.remove_accents(item)
            # map characters to integers, map space and unknown to 0:
            encoded_item = [
                self.mapping[char] if char in self.vocabulary else 0 for char in item[:min(len(item), self.maxlen)]]
            # reverse encoding
            if backwards:
                encoded_item = list(reversed(encoded_item))
            encoded_items.append(encoded_item)
        # pad sequences
        encoded_items = pad_sequences(encoded_items, maxlen=self.maxlen,
                                      truncating='post', padding='post')
        # one-hot-encoding
        x_encoded = [one_hot_encode(item, self.vocab_size) for item in encoded_items]

        return np.array(x_encoded)

    def encode_labels(self, y):
        """Encode labels into one-hot vectors:"""

        # labels in datasets are already numeric
        return to_categorical(y, num_classes=self.num_classes)

    """
    For labels that are not already numeric normalized integer
    values (e.g. text) use label encoder classes below:
    """

    def fit_label_encoder(self, y):
        self.label_encoder = LabelEncoder()
        self.label_encoder = self.label_encoder.fit(y)

    def fit_transform_text_labels(self, y):
        # integer encode
        integer_encoded = self.label_encoder.fit_transform(y)
        # binary encode
        y_encoded = to_categorical(integer_encoded, len(self.label_encoder.classes_))
        return y_encoded

    def transform_text_labels(self, y):
        # integer encode
        integer_encoded = self.label_encoder.transform(y)
        # binary encode
        y_encoded = to_categorical(integer_encoded, len(self.label_encoder.classes_))
        return y_encoded

    def inverse_transform_text_labels(self, y_encoded):
        integer_encoded = [np.argmax(y) for y in y_encoded]
        y = self.label_encoder.inverse_transform(integer_encoded)
        return y

    def save_text_label_classes(self, path):
        # save for predictions
        if hasattr(self.label_encoder, 'classes_'):
            try:
                np.save(path, self.label_encoder.classes_)
            except:
                print("Cannot save to path '{}'.".format(path))
        else:
            print("label_encoder has no attribute 'classes_' and needs to be fitted.")


def one_hot_encode(x, vocab_size):
    """One-hot encoding of integer arrays,
    leaving spaces and unknown characters all zeros"""
    # earlier approach with keras.utils.to_categorical failed due to
    # keras encoding '0' values as own class instead of all zeros
    one_hot_x = np.zeros((len(x), vocab_size), dtype=np.int)
    for idx, int_char in enumerate(x):
        if int_char > 0:
            one_hot_x[idx][int_char - 1] = 1
    return one_hot_x
