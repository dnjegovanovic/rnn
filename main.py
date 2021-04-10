from simpleRNN import simplernn
from IMDbExample import dataprepare as dp
from IMDbExample.model import models
import matplotlib as plt
import tensorflow as tf
import os
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

def train_lstm_bidirect():
    """[summary]
    """
    train_data, valid_data, test_data, tokens_count = dp.create_dataset()
    
    bi_lstm_model = models.model_lstm_bidirection(tokens_count)
    
        ## compile and train:
    bi_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy'])
    
    history = bi_lstm_model.fit(
    train_data, 
    validation_data=valid_data, 
    epochs=1)

    ## evaluate on the test data
    test_results= bi_lstm_model.evaluate(test_data)
    print('Test Acc.: {:.2f}%'.format(test_results[1]*100))

    if not os.path.exists('models'):
        os.mkdir('models')
    
    bi_lstm_model.save('models/Bidir-LSTM-full-length-seq.h5')
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    #simplernn.simplernn_eval()

    train_lstm_bidirect()


   