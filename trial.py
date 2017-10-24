"""""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import confusion_matrix


a=np.array([[0,0,0,1],[0,0,1,0],[0,0,0,1]])
max_ind=tf.argmax(a,axis=1)

a=tf.convert_to_tensor(a)
b=np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0]])
b=tf.convert_to_tensor(b)

a=math_ops.cast(a, dtype=dtypes.bool)
b=math_ops.cast(b, dtype=dtypes.bool)
a,b=confusion_matrix.remove_squeezable_dimensions(a,b)
precision,precision_op=tf.metrics.precision(a,b,name="precision_operation")
t_p=tf.metrics.true_positives(a,b)
running_vars_precision = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision_operation")
running_vars_initializer_precision = tf.variables_initializer(var_list=running_vars_precision)
#t_p=math_ops.logical_and(math_ops.equal(a, True),
                                           # math_ops.equal(b, True))
sess=tf.InteractiveSession()
print(sess.run(max_ind))

sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
print(sess.run(a))
print(sess.run(b))
print(sess.run(t_p))

#sess.run(precision_op)
#print(sess.run(precision))

"""""

