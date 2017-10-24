
import tensorflow as tf
import numpy as np
import tensorflow as tf


def attention(inputs, attention_size, return_alphas=False):





    inputs_shape = inputs.shape
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

    # Attention mechanism
    #(hidden_size,attention_size)
    W_omega = tf.get_variable(name="w_omega",shape=[hidden_size, attention_size],initializer=tf.contrib.layers.xavier_initializer())
    #(attention_size,1)
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    tf.summary.histogram("W_omega",W_omega)
    tf.summary.histogram("b_omega",b_omega)
    #(batch_size,seq_len,num_units)

    #(batch_size*seq_len,atention_size)   (batch_size*seq_len,hidden_size),(hidden_size,attention_size)
    inputs=tf.tanh(inputs)
    v = tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1])
    #(batch_size*seq_length,1)           (batch_size*seq_len,atention_size),(attention_size,1)
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    #(batch_size,seq_len)
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    #(batch_size,sequence_length)
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector

    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas