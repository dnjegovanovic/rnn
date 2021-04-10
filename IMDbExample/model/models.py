import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional


def model_simplernn():
    """[summary]

    Returns:
        [type]: [description]
    """
    model = Sequential()
    model.add(Embedding(1000, 32))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32))
    model.add(Dense(1))
    model.summary()

    return model


def model_lstm_bidirection(embedding_dim, vocab_size,
                           recurrent_type='SimpleRNN',
                           n_recurrent_units=64,
                           n_recurrent_layers=1,
                           bidirectional=True):
    """[summary]

    Args:
        embedding_dim ([type]): [description]
        vocab_size ([type]): [description]
        recurrent_type (str, optional): [description]. Defaults to 'SimpleRNN'.
        n_recurrent_units (int, optional): [description]. Defaults to 64.
        n_recurrent_layers (int, optional): [description]. Defaults to 1.
        bidirectional (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    tf.random.set_seed(1)

    # build the model
    model = tf.keras.Sequential()

    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name='embed-layer')
    )

    for i in range(n_recurrent_layers):
        return_sequences = (i < n_recurrent_layers-1)

        if recurrent_type == 'SimpleRNN':
            recurrent_layer = SimpleRNN(
                units=n_recurrent_units,
                return_sequences=return_sequences,
                name='simprnn-layer-{}'.format(i))
        elif recurrent_type == 'LSTM':
            recurrent_layer = LSTM(
                units=n_recurrent_units,
                return_sequences=return_sequences,
                name='lstm-layer-{}'.format(i))
        elif recurrent_type == 'GRU':
            recurrent_layer = GRU(
                units=n_recurrent_units,
                return_sequences=return_sequences,
                name='gru-layer-{}'.format(i))

        if bidirectional:
            recurrent_layer = Bidirectional(
                recurrent_layer, name='bidir-'+recurrent_layer.name)

        model.add(recurrent_layer)

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model
