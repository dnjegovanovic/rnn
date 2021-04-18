import numpy as np
import tensorflow as tf


def split_input_target(chunk):
    """[summary]
    Define the function for splitting x & y
    Args:
        chunk ([type]): [description]

    Returns:
        [type]: [description]
    """
    input_seq = chunk[:-1]
    output_seq = chunk[1:]
    return input_seq, output_seq


def prepare_data(path=r'.//GenerateText//data//1268-0.txt', seq_lenght=40, batch_size=64, buffer_size=10000):
    """[summary]

    Args:
        path (regexp, optional): [description]. Defaults to r'.//GenerateText//data//1268-0.txt'.
        seq_lenght (int, optional): [description]. Defaults to 40.
        batch_size (int, optional): [description]. Defaults to 64.
        buffer_size (int, optional): [description]. Defaults to 10000.
    """

    with open(path, 'r', encoding="utf8") as fp:
        text = fp.read()

    start_indx = text.find('THE MYSTERIOUS ISLAND')
    end_indx = text.find('End of the Project Gutenberg')
    text = text[start_indx:end_indx]
    char_set = set(text)

    print('Total lenght:{}'.format(len(char_set)))

    # Map charcater to int

    chars_sorted = sorted(char_set)
    char2int = {ch: i for i, ch in enumerate(chars_sorted)}
    char_array = np.array(chars_sorted)

    # take only int for evry char
    text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)

    print('Text encoded shape:{}'.format(text_encoded.shape))

    # Create Tensorfloe data set

    ds_text_encoded = tf.data.Dataset.from_tensor_slices(text_encoded)

    for ex in ds_text_encoded.take(5):
        print('{} -> {}'.format(ex.numpy(), char_array[ex.numpy()]))

    # Format text in batches

    chunk_size = seq_lenght + 1

    ds_chunks = ds_text_encoded.batch(chunk_size, drop_remainder=True)

    ds_seq = ds_chunks.map(split_input_target)

    # inspection:
    for example in ds_seq.take(2):
        print(' Input (x):', repr(''.join(char_array[example[0].numpy()])))
        print('Target (y):', repr(''.join(char_array[example[1].numpy()])))
        print()

    ds = ds_seq.shuffle(buffer_size).batch(batch_size)

    return ds
