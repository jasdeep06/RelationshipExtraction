###########################################################
#Note regarding tf.nn.bidirectional_dynamic_rnn()         #
#It takes input of shape [batch_size,max_time,depth] where#
#max_time is length of largest sequence in the batch      #
#depth is dimensionality of input vector.                 #
#It returns 2 tuples outputs and states consisting of     #
#(output_fw,output_bw) and (state_fw,state_bw)            #
#The output_fw is of shape [batch_size,max_time,num_units]#
###########################################################
"""""
#code to figure out dimensions
X = np.random.randn(2, 10, 8)  #batch,max_time,depth
print(X)
print("________________________________________")
X[1, 6, :] = 0
print(X)
X_lengths = [10, 6]

cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)
#x_new=tf.unstack(X,10,1)
outputs, states = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell,
    cell_bw=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

output_fw, output_bw = outputs
states_fw, states_bw = states
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(sess.run((output_fw)))
"""""
#importing necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

#importing dataset from tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot="True")

#hyperparameters and constants
batch_size=128
max_time=28
depth=28
num_units=128
num_classes=10
learning_rate=0.001

#placeholder for inputs and labels
X=tf.placeholder(tf.float32,[None,max_time,depth])
y=tf.placeholder(tf.int32,[None,num_classes])

#fully connected layer or hidden layer
out_weight=tf.Variable(tf.random_normal([num_units,num_classes]))
out_bais=tf.Variable(tf.random_normal([num_classes]))


#network
#forward lstm cell
lstm_fw_cell=rnn.BasicLSTMCell(num_units=num_units)
#backward lstm cell
lstm_bw_cell=rnn.BasicLSTMCell(num_units=num_units)

#outputs and states of RNN
outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,cell_bw=lstm_bw_cell,inputs=X,dtype="float32")
output_fw,output_bw=outputs
#element wise addition of forward and backward outputs
net_output=output_fw+output_bw

#output at kast time step
#dimension changed from [batch_size,max_step,num_units] to [batch_size,num_units]
last_relevant=net_output[:,-1,:]

#generating logits
logits=tf.add(tf.matmul(last_relevant,out_weight),out_bais)

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#training loop
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter=1
    while iter<400:
        batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)

        batch_x=batch_x.reshape((batch_size,max_time,depth))

        sess.run(opt, feed_dict={X: batch_x, y: batch_y})

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={X:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={X:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1


    #calculating test accuracy
    test_data = mnist.test.images[:128].reshape((-1, max_time, depth))
    test_label = mnist.test.labels[:128]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, y: test_label}))


