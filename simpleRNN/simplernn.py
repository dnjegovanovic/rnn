import tensorflow as tf

tf.random.set_seed(1)


def simplernn_eval():
    rnn_layer = tf.keras.layers.SimpleRNN(units=2, use_bias=True, return_sequences=True)

    rnn_layer.build(input_shape=(None, None, 5))

    w_xh, w_oo, b_h = rnn_layer.weights

    print("W_xh shape:{}".format(w_xh.shape))
    print("W_oo shape:{}".format(w_oo.shape))
    print("b_h shape:{}".format(b_h.shape))

    x_seq = tf.convert_to_tensor([[1.0] * 5, [2.0] * 5, [3.0] * 5], dtype=tf.float32)

    # output os SimpleRNN
    output = rnn_layer(tf.reshape(x_seq, shape=(1, 3, 5)))

    # manually computing the output
    out_man = []

    for t in range(len(x_seq)):
        xt = tf.reshape(x_seq[t], (1, 5))
        print("Time step {} =>".format(t))

        print("Input    :", xt.numpy())

        ht = tf.matmul(xt, w_xh) + b_h
        print("Hidden    :", ht.numpy())

        if t > 0:
            prev_o = out_man[t - 1]
        else:
            prev_o = tf.zeros(shape=(ht.shape))

        ot = ht + tf.matmul(prev_o, w_oo)
        ot = tf.math.tanh(ot)
        out_man.append(ot)
        print("Output (manual): {}".format(ot.numpy()))

        print("Simple RNN output:{}".format(t), output[0][t].numpy())

        print()
