import tensorflow as tf
from .scripts import dataprocessing as dp
from .scripts import dataprocessing


def sample(model, starting_str, char2int, logits, char_array,
           len_generated_text=500,
           max_input_length=40,
           scale_factor=1.0):

    encoded_input = [char2int[s] for s in starting_str]
    encoded_input = tf.reshape(encoded_input, (1, -1))

    generated_str = starting_str

    model.reset_states()
    for i in range(len_generated_text):
        logits = model(encoded_input)
        logits = tf.squeeze(logits, 0)

        scaled_logits = logits * scale_factor
        new_char_indx = tf.random.categorical(
            scaled_logits, num_samples=1)

        new_char_indx = tf.squeeze(new_char_indx)[-1].numpy()

        generated_str += str(char_array[new_char_indx])

        new_char_indx = tf.expand_dims([new_char_indx], 0)
        encoded_input = tf.concat(
            [encoded_input, new_char_indx],
            axis=1)
        encoded_input = encoded_input[:, -max_input_length:]

    return generated_str


def eval_net():
    # laod model
    model = tf.keras.models.load_model('./modelsGT/generate-text.h5')
    ds, char_array, char2int = dataprocessing.prepare_data()
    logits = [[1.0, 1.0, 3.0]]
    print('Probabilities:', tf.math.softmax(logits).numpy()[0])

    tf.random.set_seed(1)
    print(sample(model, 'The island', char2int, logits, char_array))
