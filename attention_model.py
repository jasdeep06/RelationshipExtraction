
import tensorflow as tf
import numpy as np

"""""
array=np.random.randn(2,3,4)
array1=np.random.randn(4)
tensor=tf.convert_to_tensor(array)
tensor1=tf.convert_to_tensor(array1)

reshaped=tf.reshape(tensor,[-1,4])
reshaped1=tf.reshape(tensor1,[1,-1])
sess=tf.InteractiveSession()
print(sess.run(tensor))
print("__________________")
op=sess.run(reshaped)
print(op)
print(sess.run(tf.shape(op)))
print("+++++++++++++++++++++++++++")
print(sess.run(tensor1))
print("__________________")
op1=sess.run(reshaped1)
print(op1)
print(sess.run(tf.shape(op1)))

"""""
array=np.random.randn(2,3,4)
array1=np.random.randn(4,6)
#3,4
array2=array[0,:,:]
#3,4
array3=array[1,:,:]
#4,3
array4=array1[:,0:3]
#4,3
array5=array1[:,3:]
print(array)
print("_____________________")

print(array2)
print("_____________________")
print(array3)
print("_____________________")
print(array1)
print("_____________________")
print(array4)
print("_____________________")
print(array5)

array6=np.random.rand(6,5)
array7=np.random.rand(1,5)
print("_____________________")

print(array6)
print("_____________________")

print(array7)
print("_____________________")


tensor=tf.convert_to_tensor(array)
tensor1=tf.convert_to_tensor(array1)
tensor2=tf.convert_to_tensor(array2)
tensor3=tf.convert_to_tensor(array3)
tensor4=tf.convert_to_tensor(array4)
tensor5=tf.convert_to_tensor(array5)
tensor6=tf.convert_to_tensor(array6)
tensor7=tf.convert_to_tensor(array7)




reshaped=tf.reshape(tensor,[-1,4])
result=tf.matmul(reshaped,tensor1)
result1=tf.matmul(tensor2,tensor1)
result2=tf.matmul(tensor3,tensor1)
result3=tf.multiply(array6,array7)

sess=tf.InteractiveSession()
print(sess.run(result))
print(sess.run(result1))
print(sess.run(result2))
print(sess.run(result3))