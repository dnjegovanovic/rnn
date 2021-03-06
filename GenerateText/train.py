import os
import tensorflow as tf
import matplotlib.pyplot as plt
from .scripts import dataprocessing
from .models import model


def train():
    ds, char_array, char2int = dataprocessing.prepare_data()

    charset_size = len(char_array)
    embedding_dim = 256
    rnn_units = 512

    tf.random.set_seed(1)

    modelgt = model.build_model(
        vocab_size=charset_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)

    modelgt.summary()

    modelgt.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ))

    history = modelgt.fit(ds, epochs=10)
    if not os.path.exists('modelsGT'):
        os.mkdir('modelsGT')

    modelgt.save('modelsGT/generate-text.h5')

    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    # plt.xlabel('val_loss')
    #plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('generate_text_test_01.png')
