import tensorflow as tf


def build_model(vocab_size, embedding_dim, rnn_units):
    """[summary]

    Args:
        vocab_size ([type]): [description]
        embedding_dim ([type]): [description]
        rnn_units ([type]): [description]

    Returns:
        [type]: [description]
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(
            rnn_units, return_sequences=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model
