import tensorflow as tf
from tensorflow.contrib import rnn
from data_preprocessing import next_batch,dev_set
import numpy as np

num_units=128
embedding_size=50
vocab_size=21594
num_classes=19
learning_rate=.01
batch_size=32
max_length=112


#placeholders
sentence_vectors=tf.placeholder(tf.int64,[None,max_length])
label_vector=tf.placeholder(tf.int64,[None,num_classes])
seq_lengths=tf.placeholder(tf.int64,[None])


attention_weight=tf.get_variable(name="attention_weight",shape=[num_units,num_units],initializer=tf.random_normal_initializer)
attention_bias=tf.get_variable(name="attention_bias",shape=[num_units],initializer=tf.random_normal_initializer)

out_weight=tf.get_variable(name="out_weight",shape=[num_units,num_classes],initializer=tf.random_normal_initializer)
out_bias=tf.get_variable(name="out_bias",shape=[num_classes],initializer=tf.random_normal_initializer)

embedding=tf.get_variable(name="embedding",shape=[vocab_size,embedding_size])
input=tf.nn.embedding_lookup(embedding,sentence_vectors)


cell_fw=rnn.LSTMCell(num_units=num_units,use_peepholes=True)
cell_bw=rnn.LSTMCell(num_units=num_units,use_peepholes=True)

outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=input,sequence_length=seq_lengths,dtype="float32")

output_fw,output_bw=outputs
#(batch_size,seq_length,num_units)
net_output=output_fw+output_bw




M=tf.tanh(net_output)
#(batch_size*sequence_length,num_units)
reshaped_M=tf.reshape(M,[-1,num_units])

forward_prop_M=tf.add(tf.matmul(reshaped_M,attention_weight),attention_bias)
#(batch_size*sequence_length,num_units)
alpha=tf.nn.softmax(forward_prop_M)

reshaped_alpha=tf.reshape(alpha,[batch_size,max_length,num_units])


r=tf.multiply(net_output,reshaped_alpha)
#batch_size,num_units
r_net=tf.reduce_sum(r,1)

logits=tf.add(tf.matmul(r_net,out_weight),out_bias)



#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label_vector))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


#recall,recall_op=tf.metrics.recall(label_vector,logits)
#f1=(2*(precision)*(recall))/(precision+recall)
#model evaluation

prediction=tf.argmax(logits,1)
labels=tf.argmax(label_vector,1)+1

precision,precision_op=tf.metrics.precision(labels,prediction,name="precision_operation")
recall,recall_op=tf.metrics.recall(labels,prediction,name="recall_operation")
running_vars_precision = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision_operation")
running_vars_recall = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall_operation")
f1_score=(2*(precision)*recall)/(precision+recall)


    # Define initializer to initialize/reset running variables
running_vars_initializer_precision = tf.variables_initializer(var_list=running_vars_precision)
running_vars_initializer_recall = tf.variables_initializer(var_list=running_vars_recall)

#correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(label_vector,1))
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#training loop
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #sess.run(tf.local_variables_initializer())
    batch_number=0
    iter=1
    while iter<100:
        sess.run(running_vars_initializer_precision)
        sess.run(running_vars_initializer_recall)


        x,y,seq,batch_number=next_batch(batch_number,batch_size,"train")

        print(sess.run(prediction, feed_dict={sentence_vectors: x, label_vector: y, seq_lengths: seq}))
        print(sess.run(labels, feed_dict={sentence_vectors: x, label_vector: y, seq_lengths: seq}))

        #print(sess.run(prediction,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq}))
        #print(sess.run(labels,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq}))


        sess.run(opt, feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq})
        #print(sess.run(tf.shape(sess.run(trial_output, feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq}))))
        sess.run(precision_op,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq})
        sess.run(recall_op,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq})

        if iter %10==0:

            acc=sess.run(f1_score,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq})
            los=sess.run(loss,feed_dict={sentence_vectors: x, label_vector: y,seq_lengths:seq})
            print("For iter ",iter)
            print("F1 score ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1


    dev_batch_number=0
    accu=[]
    for i in range(30):

        dev_vec, dev_labels, dev_seq_length,dev_batch_number =next_batch(dev_batch_number,batch_size,"dev")

        accu.append(sess.run(accuracy,feed_dict={sentence_vectors: dev_vec, label_vector: dev_labels,seq_lengths:dev_seq_length}))
        print(accu)
    print(np.mean(accu))


