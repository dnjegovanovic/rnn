import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import os
import gzip
import shutil
from collections import Counter


class DataProcessing():

    def __init__(self):
        self.encoder = None
        self.max_seq_length = 100
        # r"./dataset/movie_data.csv"
        self.path_to_data = r"./IMDbExample/dataset/movie_data.csv"

    def read_data(self):
        """[summary]

        Args:
            pathToData (regexp, optional): [description]. Defaults to r"./dataset/movie_data.csv".

        Returns:
            [type]: [description]
        """
        df = pd.read_csv(self.path_to_data, encoding="utf-8")
        return df

    def encode(self, text_tensor, label):
        """[summary]
        Define the function for transformation
        Args:
            text_tensor ([type]): [description]
            label ([type]): [description]

        Returns:
            [type]: [description]
        """
        text = text_tensor.numpy()[0]
        encoded_text = self.encoder.encode(text)
        if self.max_seq_length is not None:
            encoded_text = encoded_text[-self.max_seq_length:]
        return encoded_text, label

    def encode_map_fn(self, text, label):
        """[summary]
        Wrap the encode function to a TF Op.
        Args:
            text ([type]): [description]
            label ([type]): [description]

        Returns:
            [type]: [description]
        """
        return tf.py_function(self.encode, inp=[text, label],
                              Tout=(tf.int64, tf.int64))

    def load_data(self):
        df = self.read_data()

        df.tail()

        # Step 1: Create a dataset

        target = df.pop("sentiment")

        ds_raw = tf.data.Dataset.from_tensor_slices((df.values, target.values))

        # inspection:
        for ex in ds_raw.take(3):
            tf.print(ex[0].numpy()[0][:50], ex[1])

        tf.random.set_seed(1)

        ds_raw = ds_raw.shuffle(
            50000, reshuffle_each_iteration=False)

        self.ds_raw_test = ds_raw.take(25000)
        ds_raw_train_valid = ds_raw.skip(25000)
        self.ds_raw_train = ds_raw_train_valid.take(20000)
        self.ds_raw_valid = ds_raw_train_valid.skip(20000)

    def create_dataset(self, max_seq_length=None,
                       batch_size=32):
        """[summary]

        Args:
            ds_raw_train ([type]): [description]
            ds_raw_valid ([type]): [description]
            ds_raw_test ([type]): [description]
            max_seq_length ([type], optional): [description]. Defaults to None.
            batch_size (int, optional): [description]. Defaults to 32.

        Returns:
            [type]: [description]
        """

        self.load_data()

        # Step 2: find unique words

        tokenizer = tfds.deprecated.text.Tokenizer()
        token_counts = Counter()

        for example in self.ds_raw_train:
            tokens = tokenizer.tokenize(example[0].numpy()[0])
            if self.max_seq_length is not None:
                tokens = tokens[-self.max_seq_length:]
            token_counts.update(tokens)

        print('Vocab-size:', len(token_counts))

        # Step 3: endoding each unique token into integers
        self.encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)

        example_str = 'This is an example!'
        self.encoder.encode(example_str)

        ds_train = self.ds_raw_train.map(self.encode_map_fn)
        ds_valid = self.ds_raw_valid.map(self.encode_map_fn)
        ds_test = self.ds_raw_test.map(self.encode_map_fn)

        train_data = ds_train.padded_batch(32, padded_shapes=([-1], []))
        valid_data = ds_valid.padded_batch(32, padded_shapes=([-1], []))
        test_data = ds_test.padded_batch(32, padded_shapes=([-1], []))

        return train_data, valid_data, test_data, token_counts
