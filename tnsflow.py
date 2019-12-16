import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_v2_behavior()



W = tf.placeholder([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.compat.v2.placeholder(tf.float32)



linear_model = W * x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))