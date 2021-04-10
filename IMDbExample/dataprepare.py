import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import os
import gzip
import shutil
from collections import Counter


def read_data(pathToData=r"./dataset/movie_data.csv"):
    """[summary]

    Args:
        pathToData (regexp, optional): [description]. Defaults to r"./dataset/movie_data.csv".

    Returns:
        [type]: [description]
    """
    df = pd.read_csv(pathToData, encoding="utf-8")
    return df

def encode(text_tensor, label):
    """[summary]
    Define the function for transformation
    Args:
        text_tensor ([type]): [description]
        label ([type]): [description]

    Returns:
        [type]: [description]
    """
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label


def encode_map_fn(text, label):
    """[summary]
    Wrap the encode function to a TF Op.
    Args:
        text ([type]): [description]
        label ([type]): [description]

    Returns:
        [type]: [description]
    """
    return tf.py_function(encode, inp=[text, label], 
                          Tout=(tf.int64, tf.int64))

def create_dataset():
    df = read_data("./IMDbExample/dataset/movie_data.csv")

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

    ds_raw_test = ds_raw.take(25000)
    ds_raw_train_valid = ds_raw.skip(25000)
    ds_raw_train = ds_raw_train_valid.take(20000)
    ds_raw_valid = ds_raw_train_valid.skip(20000)

    # Step 2: find unique words

    tokenizer = tfds.deprecated.text.Tokenizer()
    token_counts = Counter()

    for example in ds_raw_train:
        tokens = tokenizer.tokenize(example[0].numpy()[0])
        # t = tf.keras.preprocessing.text.text_to_word_sequence(example[0].numpy()[0])
        token_counts.update(tokens)

    print('Vocab-size:', len(token_counts))
    
    ## Step 3: endoding each unique token into integers
    global encoder
    encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)

    example_str = 'This is an example!'
    encoder.encode(example_str)
    
    ds_train = ds_raw_train.map(encode_map_fn)
    ds_valid = ds_raw_valid.map(encode_map_fn)
    ds_test = ds_raw_test.map(encode_map_fn)

    tf.random.set_seed(1)
    for example in ds_train.shuffle(1000).take(5):
        print('Sequence length:', example[0].shape)
        
    train_data = ds_train.padded_batch(32, padded_shapse=([-1],[]))
    valid_data = ds_valid.padded_batch(32, padded_shapse=([-1],[]))
    test_data = ds_test.padded_batch(32, padded_shapse=([-1],[]))
    
    return train_data, valid_data, test_data
