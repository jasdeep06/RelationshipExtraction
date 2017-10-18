import tensorflow as tf
from tensorflow.contrib import rnn
from data_preprocessing import next_batch

num_units=128
embedding_size=50
vocab_size=21593
num_classes=19
learning_rate=.01
batch_size=32
max_length=112


#placeholders
sentence_vectors=tf.placeholder(tf.int64,[None,max_length])
label_vector=tf.placeholder(tf.int64,[None,num_classes])
seq_lengths=tf.placeholder(tf.int64,[None])


embedding=tf.get_variable(name="embedding",shape=[vocab_size,embedding_size])
input=tf.nn.embedding_lookup(embedding,sentence_vectors)


cell_fw=rnn.LSTMCell(num_units=num_units,use_peepholes=True)
cell_bw=rnn.LSTMCell(num_units=num_units,use_peepholes=True)

outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=input,sequence_length=seq_lengths,dtype="float32")

output_fw,output_bw=outputs

net_output=output_fw+output_bw

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
X,y,seq,num=next_batch(0,10)
print(type(X))
print(type(y))
print(type(seq))
print(sess.run(net_output,feed_dict={sentence_vectors:X,seq_lengths:seq}))




