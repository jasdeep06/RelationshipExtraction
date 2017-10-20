import numpy as np
import tensorflow as tf

a=np.array([1,2,3,4,5,1,2,6,7,4])
a=tf.convert_to_tensor(a)
b=np.array([0,0,0,0,0,0,0,0,0,0])
b=tf.convert_to_tensor(b)
precision,precision_op=tf.metrics.precision(a,b,name="precision_operation")
running_vars_precision = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision_operation")
running_vars_initializer_precision = tf.variables_initializer(var_list=running_vars_precision)

sess=tf.InteractiveSession()
sess.run(running_vars_initializer_precision)
sess.run(tf.global_variables_initializer())
sess.run(precision_op)
print(sess.run(precision))