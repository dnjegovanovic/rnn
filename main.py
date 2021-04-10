from simpleRNN import simplernn
from IMDbExample import dataprepare as dp
from IMDbExample.model import models
import matplotlib.pyplot as plt
import tensorflow as tf
import os
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"


def train_lstm_bidirect():

    batch_size = 32
    embedding_dim = 20
    max_seq_length = 100

    data = dp.DataProcessing()
    data.max_seq_length = max_seq_length
    train_data, valid_data, test_data, tokens_count = data.create_dataset()

    vocab_size = len(tokens_count)+2
    bi_lstm_model = models.model_lstm_bidirection(embedding_dim, vocab_size,
                                                  recurrent_type='SimpleRNN',
                                                  n_recurrent_units=64,
                                                  n_recurrent_layers=1,
                                                  bidirectional=True)

    bi_lstm_model.summary()

    # compile and train:
    bi_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                          loss=tf.keras.losses.BinaryCrossentropy(
                              from_logits=False),
                          metrics=['accuracy'])

    history = bi_lstm_model.fit(
        train_data,
        validation_data=valid_data,
        epochs=10)

    # evaluate on the test data
    test_results = bi_lstm_model.evaluate(test_data)
    print('Test Acc.: {:.2f}%'.format(test_results[1]*100))

    if not os.path.exists('models'):
        os.mkdir('models')

    bi_lstm_model.save('models/Bidir-LSTM-full-length-seq.h5')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('val_loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('rnn_test_01.png')
    # plt.show()


if __name__ == "__main__":
    train_lstm_bidirect()
