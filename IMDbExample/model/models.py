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

def model_lstm_bidirection(token_counts):
    
    embedding_dim = 20
    vocab_size = len(token_counts) + 2

    tf.random.set_seed(1)
    ## build the model
    bi_lstm_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name='embed-layer'),
        
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, name='lstm-layer'),
            name='bidir-lstm'), 

        tf.keras.layers.Dense(64, activation='relu'),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    bi_lstm_model.summary()
    
    return bi_lstm_model