import tensorflow as tf

with tf.Session() as sess:
  with tf.device("/gpu:1"):
     ma1 = tf.constant([[3., 3.]])
     ma2 = tf.constant([[2.],[2.]])
     res = tf.matmul(ma1, ma2)

     print(sess.run(res))
