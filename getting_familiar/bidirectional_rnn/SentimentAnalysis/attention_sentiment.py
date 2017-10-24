import tensorflow as tf
from utils import next_batch,label_to_one_hot,log_params
from tensorflow.contrib import rnn
from attention import attention


#parameters and hyperparameters
max_length=45
vocab_size=20306
embedding_size=50
num_units=128
num_classes=3
learning_rate=0.001
batch_size=64


#placeholders
#[batch_size,max_length]
tweet_vec=tf.placeholder(tf.int64,[None,max_length])
#[batch_size,num_classes] (in form of one hot)
tweet_label=tf.placeholder(tf.int64,[None,num_classes])
#[batch_size]
seq_length=tf.placeholder(tf.int32,[None])

#fully connected layer or hidden layer
#[num_units,num_classes]
out_weight=tf.Variable(tf.random_normal([num_units,num_classes]))
#[num_classes]
out_bais=tf.Variable(tf.random_normal([num_classes]))

#[vocab_size,embedding_size]
embedding=tf.get_variable(name="embedding",shape=[vocab_size,embedding_size])

tf.summary.histogram("embedding_summary",embedding)
#tf.summary.histogram("attention_weight_summary",attention_weight)
#tf.summary.histogram("attention_bias_summary",attention_bias)
tf.summary.histogram("out_weight_summary",out_weight)
tf.summary.histogram("out_bias_summary",out_bais)

#[batch_size,max_length,embedding_size]
input=tf.nn.embedding_lookup(embedding,tweet_vec)


cell_fw=rnn.BasicLSTMCell(num_units=num_units)
cell_bw=rnn.BasicLSTMCell(num_units=num_units)


outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=input,sequence_length=seq_length,dtype="float32")

#net_output=tf.concat(outputs,2)
#[batch_size,max_length,num_units]
output_fw,output_bw=outputs
#[batch_size,max_length,num_units]
net_output=output_fw+output_bw

"""""
#to figure out the last relevent output
size_of_batch = tf.shape(net_output)[0]
max_length = tf.shape(net_output)[1]
out_size = int(net_output.get_shape()[2])
index = tf.range(0, size_of_batch) * max_length + (seq_length - 1)
flat = tf.reshape(net_output, [-1, out_size])

#[batch_size,num_units]
relevant_output = tf.gather(flat, index)



#generating logits
logits=tf.add(tf.matmul(relevant_output,out_weight),out_bais)

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tweet_label))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


prediction=tf.argmax(logits,1)
label=tf.argmax(tweet_label,1)
#model evaluation
correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(tweet_label,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#training loop
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batch_number=0
    iter=1
    while iter<500:
        x,y,seq,batch_number=next_batch(batch_number,batch_size,"train")
        y=label_to_one_hot(y)
        print(sess.run(prediction,feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq}))
        print(sess.run(label,feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq}))



        sess.run(opt, feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq})
        #print(sess.run(tf.shape(sess.run(trial_output, feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq}))))

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq})
            los=sess.run(loss,feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1



    test_batch_number=0
    test_x,test_y,test_seq_length,__=next_batch(test_batch_number,2200,"test")
    test_y=label_to_one_hot(test_y)
    print(sess.run(accuracy,feed_dict={tweet_vec:test_x,tweet_label:test_y,seq_length:test_seq_length}))



"""""
r_net,alpha=attention(net_output,50,True)
logits=tf.add(tf.matmul(r_net,out_weight),out_bais)



#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tweet_label))
tf.summary.scalar("loss",loss)

#optimization
with tf.name_scope("train"):
    opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


#recall,recall_op=tf.metrics.recall(label_vector,logits)
#f1=(2*(precision)*(recall))/(precision+recall)
#model evaluation
max_indices=tf.argmax(logits,axis=1)
one_hot_logits=tf.one_hot(indices=max_indices,depth=3,dtype=tf.int64)

precision,precision_op=tf.metrics.precision(tweet_label,one_hot_logits,name="precision_operation")
recall,recall_op=tf.metrics.recall(tweet_label,one_hot_logits,name="recall_operation")
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
    tf_dir="tensorboard_analysis/learning_rate_variation/"+"learning_rate_"+str(learning_rate)

    writer=tf.summary.FileWriter(tf_dir)
    writer.add_graph(sess.graph)
    #sess.run(tf.local_variables_initializer())
    num_epochs=0
    batch_number=0
    iteration=1
    while num_epochs<13:
        sess.run(running_vars_initializer_precision)
        sess.run(running_vars_initializer_recall)
        #sess.run(tf.local_variables_initializer())


        x,y,seq,batch_number,num_epochs=next_batch(batch_number,batch_size,num_epochs,"train")
        y=label_to_one_hot(y)


        sess.run(opt, feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq})
        #print(sess.run(tf.shape(sess.run(trial_output, feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq}))))
        sess.run(precision_op,feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq})
        sess.run(recall_op,feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq})

        if iteration %10==0:

            acc=sess.run(f1_score,feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq})
            los=sess.run(loss,feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq})
            s = sess.run(merge_summary, feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq})
            writer.add_summary(s, iteration)
            print("For iter ",iteration)
            print("F1 score ",acc)
            print("Loss ",los)
            print("__________________")

        iteration=iteration+1

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
    test_num_epochs=0
    #while test_iter<31:
    x_test,y_test,seq_len_test,test_batch_number,test_num_epochs=next_batch(test_batch_number,2200,test_num_epochs,"dev")
    y_test=label_to_one_hot(y_test)

    #print(sess.run(f1_score,feed_dict={sentence_vectors:x_test,label_vector:y_test,seq_lengths:seq_len_test}))
    sess.run(precision_op, feed_dict={tweet_vec:x_test,tweet_label:y_test,seq_length:seq_len_test})
    sess.run(recall_op, feed_dict={tweet_vec:x_test,tweet_label:y_test,seq_length:seq_len_test})
    #test_iter=test_iter+1
    score=sess.run(f1_score)
    print(score)
    param_string="max_length "+str(max_length)+" vocab_size "+str(vocab_size)+" embedding_size "+str(embedding_size)+" num_units "+str(num_units)+" num_classes "+str(num_classes)+" learning rate "+str(learning_rate)+" batch_size "+str(batch_size)+" iterations "+str(iteration) + " num_epoch "+str(num_epochs)
    final_string=param_string+" has f1 score of " + str(score)
    log_params("log_dir/logging.txt",final_string)
    log_params("tensorboard_analysis/learning_rate_variation/rest_of_params.txt",param_string)
