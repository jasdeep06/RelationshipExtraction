import tensorflow as tf
from tensorflow.contrib import rnn
from data_preprocessing import next_batch,dev_set
from attention_model import attention
import numpy as np

num_units=128
embedding_size=50
vocab_size=21594
num_classes=19
learning_rate=.01
batch_size=32
max_length=112


#placeholders
with tf.name_scope("Placeholders"):
    sentence_vectors=tf.placeholder(tf.int64,[None,max_length],name="sentence_placeholder")
    label_vector=tf.placeholder(tf.int64,[None,num_classes],name="label_placeholder")
    seq_lengths=tf.placeholder(tf.int64,[None],name="seq_length_placeholder")

with tf.name_scope("Attention_weight_and_bias"):
    attention_weight=tf.get_variable(name="attention_weight",shape=[num_units,num_units],initializer=tf.contrib.layers.xavier_initializer())
    #attention_weight=tf.get_variable(name="attention_weight",shape=[num_units,1],initializer=tf.contrib.layers.xavier_initializer())

    attention_bias=tf.get_variable(name="attention_bias",shape=[num_units],initializer=tf.zeros_initializer)
    #attention_bias=tf.get_variable(name="attention_bias",shape=[1],initializer=tf.zeros_initializer)

with tf.name_scope("FC_Layer_weight_and_bias"):
    out_weight=tf.get_variable(name="out_weight",shape=[num_units,num_classes],initializer=tf.contrib.layers.xavier_initializer())
    out_bias=tf.get_variable(name="out_bias",shape=[num_classes],initializer=tf.zeros_initializer)

embedding=tf.get_variable(name="embedding",shape=[vocab_size,embedding_size],initializer=tf.contrib.layers.xavier_initializer())
input=tf.nn.embedding_lookup(embedding,sentence_vectors)

tf.summary.histogram("embedding_summary",embedding)
#tf.summary.histogram("attention_weight_summary",attention_weight)
#tf.summary.histogram("attention_bias_summary",attention_bias)
tf.summary.histogram("out_weight_summary",out_weight)
tf.summary.histogram("out_bias_summary",out_bias)


cell_fw=rnn.LSTMCell(num_units=num_units,use_peepholes=True)
cell_bw=rnn.LSTMCell(num_units=num_units,use_peepholes=True)

outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=input,sequence_length=seq_lengths,dtype="float32")

output_fw,output_bw=outputs
#(batch_size,seq_length,num_units)
net_output=output_fw+output_bw


"""""

M=tf.tanh(net_output)
#(batch_size*sequence_length,num_units)
reshaped_M=tf.reshape(M,[-1,num_units])

forward_prop_M=tf.add(tf.matmul(reshaped_M,attention_weight),attention_bias)
#(batch_size*sequence_length,num_units)
alpha=tf.nn.softmax(forward_prop_M)

reshaped_alpha=tf.reshape(alpha,[batch_size,max_length,num_units])
#reshaped_alpha=tf.reshape(alpha,[-1,max_length,1])


r=tf.multiply(net_output,reshaped_alpha)


#batch_size,num_units
r_net=tf.reduce_sum(r,1)

"""""
r_net,alpha=attention(net_output,50,True)
logits=tf.add(tf.matmul(r_net,out_weight),out_bias)



#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label_vector))
tf.summary.scalar("loss",loss)

#optimization
with tf.name_scope("train"):
    opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


#recall,recall_op=tf.metrics.recall(label_vector,logits)
#f1=(2*(precision)*(recall))/(precision+recall)
#model evaluation
max_indices=tf.argmax(logits,axis=1)
one_hot_logits=tf.one_hot(indices=max_indices,depth=19,dtype=tf.int64)

precision,precision_op=tf.metrics.precision(label_vector,one_hot_logits,name="precision_operation")
recall,recall_op=tf.metrics.recall(label_vector,one_hot_logits,name="recall_operation")
running_vars_precision = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision_operation")
running_vars_recall = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall_operation")
f1_score=(2*(precision)*recall)/(precision+recall)
tf.summary.scalar("f1 score",f1_score)



    # Define initializer to initialize/reset running variables
running_vars_initializer_precision = tf.variables_initializer(var_list=running_vars_precision)
running_vars_initializer_recall = tf.variables_initializer(var_list=running_vars_recall)

#correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(label_vector,1))
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#training loop
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    merge_summary=tf.summary.merge_all()
    writer=tf.summary.FileWriter("tensorboard_analysis/trial")
    writer.add_graph(sess.graph)
    #sess.run(tf.local_variables_initializer())
    batch_number=0
    iter=1
    while iter<1000:
        sess.run(running_vars_initializer_precision)
        sess.run(running_vars_initializer_recall)
        #sess.run(tf.local_variables_initializer())


        x,y,seq,batch_number=next_batch(batch_number,batch_size,"train")
        #print(sess.run(alpha, feed_dict={sentence_vectors: x, label_vector: y, seq_lengths: seq}))
        #print(sess.run(tf.shape(sess.run(forward_prop_M,feed_dict={sentence_vectors: x, label_vector: y, seq_lengths: seq}))))
        #print(sess.run(label_vector, feed_dict={sentence_vectors: x, label_vector: y, seq_lengths: seq}))
        #print(sess.run(logits, feed_dict={sentence_vectors: x, label_vector: y, seq_lengths: seq}))
        #print(sess.run(one_hot_logits, feed_dict={sentence_vectors: x, label_vector: y, seq_lengths: seq}))


        #print(sess.run(prediction,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq}))
        #print(sess.run(labels,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq}))
        #print(sess.run(max_indices,feed_dict={sentence_vectors: x, label_vector: y, seq_lengths: seq}))
        #print(sess.run(one_hot_logits,feed_dict={sentence_vectors: x, label_vector: y, seq_lengths: seq}))

        sess.run(opt, feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq})
        #print(sess.run(tf.shape(sess.run(trial_output, feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq}))))
        sess.run(precision_op,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq})
        sess.run(recall_op,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq})


        if iter %10==0:

            acc=sess.run(f1_score,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq})
            los=sess.run(loss,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq})
            s = sess.run(merge_summary, feed_dict={sentence_vectors: x, label_vector: y, seq_lengths: seq})
            writer.add_summary(s, iter)
            print("For iter ",iter)
            print("F1 score ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1

    """""
    dev_batch_number=0
    accu=[]
    for i in range(30):

        dev_vec, dev_labels, dev_seq_length,dev_batch_number =next_batch(dev_batch_number,batch_size,"dev")

        accu.append(sess.run(accuracy,feed_dict={sentence_vectors: dev_vec, label_vector: dev_labels,seq_lengths:dev_seq_length}))
        print(accu)
    print(np.mean(accu))
    """""

    sess.run(running_vars_initializer_precision)
    sess.run(running_vars_initializer_recall)
    test_iter=0
    test_batch_number=0
    #while test_iter<31:
    x_test,y_test,seq_len_test,test_batch_number=next_batch(test_batch_number,950,"dev")
    #print(sess.run(f1_score,feed_dict={sentence_vectors:x_test,label_vector:y_test,seq_lengths:seq_len_test}))
    sess.run(precision_op, feed_dict={sentence_vectors:x_test,label_vector:y_test,seq_lengths:seq_len_test})
    sess.run(recall_op, feed_dict={sentence_vectors:x_test,label_vector:y_test,seq_lengths:seq_len_test})
    #test_iter=test_iter+1
    print(sess.run(f1_score))