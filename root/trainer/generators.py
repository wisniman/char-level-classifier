# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
# import math
import tensorflow as tf
from trainer.tools import threadsafe_generator
from keras.utils import Sequence


class DataGenerator(object):
    """ Data generator for training and evaluation in Keras """

    def __init__(self, input_file, label_column, data_columns, encoder, batch_size, shuffle=False):
        """Initialization"""
        self.input_file = input_file
        self.label_column = label_column
        self.data_columns = data_columns
        self.encoder = encoder
        self.batch_size = batch_size
        self.shuffle = shuffle

    @threadsafe_generator
    def generate(self, backwards=False):
        """Generate batches of data"""

        # infinite Loop
        while True:
            # generate data
            (data, labels) = self.__data_generation()

            # shuffle data
            if self.shuffle:
                (data, labels) = self.__shuffle_data(data, labels)

            # generate batches
            for idx in range(0, len(data) - (len(data) % self.batch_size), self.batch_size):
                data_batch = data[idx:idx + self.batch_size]
                labels_batch = labels[idx:idx + self.batch_size]
                # Encode data:
                encoded_data_batch = self.encoder.encode_data(data_batch, backwards=backwards)
                encoded_labels_batch = self.encoder.encode_labels(labels_batch)
                # Output of generator must be a tuple of either 2 or 3 numpy arrays :
                # (X, y) or (X, y, sample_weight)
                yield(encoded_data_batch, encoded_labels_batch)

    def __data_generation(self, input_data):
        """load csv, get data & labels from csv-dataframe"""

        input_data = pd.read_csv(tf.gfile.Open(self.input_file[0]),
                                 encoding='utf-8',
                                 header=None)
        input_data = input_data.dropna()
        # retrieve data from data columns
        data = input_data[self.data_columns[0]]
        for i in range(1, len(self.data_columns)):
            data += '\n' + input_data[self.data_columns[i]]
        data = np.array(data)
        # retrieve labels from label column
        labels = input_data[self.label_column] - 1
        labels = np.array(labels)
        # return data & labels
        return (data, labels)

    def __shuffle_data(self, x, y):
        """shuffle dataset"""

        stacked = np.stack((x, y), axis=1)
        np.random.shuffle(stacked)
        x_shuffled = np.array(stacked[:, 0]).flatten()
        y_shuffled = np.array(stacked[:, 1:]).flatten()
        return (x_shuffled, y_shuffled)


class DataSequence(Sequence):

    def __init__(self, input_file, label_column, data_columns, encoder, backwards, batch_size, steps_per_epoch, shuffle=False):
        """Initialization"""

        self.input_file = input_file
        self.label_column = label_column
        self.data_columns = data_columns
        self.encoder = encoder
        self.backwards = backwards
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.shuffle = shuffle
        self.data, self.labels = self.__data_generation()

    def __len__(self):
        """retrun number of steps per epoch"""

        # has to be initialized bc keras overwrites the 'step_per_epoch'
        # agument of the generator with this value as of version 2.0.9
        return self.steps_per_epoch

    def __getitem__(self, idx):
        """retrieve batches of items"""

        data_batch = self.data[self.batch_size * idx:self.batch_size * (idx + 1)]
        labels_batch = self.labels[self.batch_size * idx:self.batch_size * (idx + 1)]
        # Encode data:
        encoded_data_batch = self.encoder.encode_data(data_batch, backwards=self.backwards)
        encoded_labels_batch = self.encoder.encode_labels(labels_batch)
        # Output of generator must be a tuple of either 2 or 3 numpy arrays :
        # (X, y) or (X, y, sample_weight)
        return (encoded_data_batch, encoded_labels_batch)
        # return encoded_data_batch, encoded_labels_batch

    def __data_generation(self):
        """load csv, get data & labels from csv-dataframe"""

        input_data = pd.read_csv(tf.gfile.Open(self.input_file[0]),
                                 encoding='utf-8',
                                 header=None)
        input_data = input_data.dropna()
        # retrieve data from data columns
        data = input_data[self.data_columns[0]]
        for i in range(1, len(self.data_columns)):
            data += '\n' + input_data[self.data_columns[i]]
        data = np.array(data)
        # retrieve labels from label column
        labels = input_data[self.label_column] - 1
        labels = np.array(labels)
        # return data & labels
        return data, labels

    def __shuffle_data(self, x, y):
        """shuffle dataset"""
        stacked = np.stack((x, y), axis=1)
        np.random.shuffle(stacked)
        x_shuffled = np.array(stacked[:, 0]).flatten()
        y_shuffled = np.array(stacked[:, 1:]).flatten()
        return (x_shuffled, y_shuffled)

    def on_epoch_end(self):
        """perform shuffle ath the end of each epoch"""

        (self.data, self.labels) = self.__shuffle_data(self.data, self.labels)
