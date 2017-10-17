import tensorflow as tf
from utils import next_batch,label_to_one_hot
from tensorflow.contrib import rnn


#parameters and hyperparameters
max_length=45
vocab_size=20306
embedding_size=50
num_units=128
num_classes=3
learning_rate=0.01
batch_size=32


#placeholders
tweet_vec=tf.placeholder(tf.int64,[None,max_length])
tweet_label=tf.placeholder(tf.int64,[None,num_classes])
seq_length=tf.placeholder(tf.int32,[None])

#fully connected layer or hidden layer
out_weight=tf.Variable(tf.random_normal([num_units,num_classes]))
out_bais=tf.Variable(tf.random_normal([num_classes]))

embedding=tf.get_variable(name="embedding",shape=[vocab_size,embedding_size])

input=tf.nn.embedding_lookup(embedding,tweet_vec)


cell_fw=rnn.BasicLSTMCell(num_units=num_units)
cell_bw=rnn.BasicLSTMCell(num_units=num_units)

outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=input,sequence_length=seq_length,dtype="float32")

output_fw,output_bw=outputs

net_output=output_fw+output_bw

#to figure out the last relevent output
size_of_batch = tf.shape(net_output)[0]
max_length = tf.shape(net_output)[1]
out_size = int(net_output.get_shape()[2])
index = tf.range(0, size_of_batch) * max_length + (seq_length - 1)
flat = tf.reshape(net_output, [-1, out_size])
relevant_output = tf.gather(flat, index)



#generating logits
logits=tf.add(tf.matmul(relevant_output,out_weight),out_bais)

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tweet_label))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(tweet_label,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#training loop
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batch_number=0
    iter=1
    while iter<1000:
        x,y,seq,batch_number=next_batch(batch_number,batch_size,"train")
        y=label_to_one_hot(y)


        sess.run(opt, feed_dict={tweet_vec: x, tweet_label: y,seq_length:seq})

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




