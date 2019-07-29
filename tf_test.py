/*tensorflow的简单的回归和分类代码*/

import tensorflow as tf 
import numpy as np 

x = np.random.normal(1,0.1,[5,10])
# y = np.mean(3 * x + 0.5,-1)
y = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0]])
inputs = tf.placeholder(shape=[None,10],dtype=tf.float32)
gt = tf.placeholder(shape=[None,3],dtype=tf.float32)

W = tf.Variable(tf.random_normal(shape=[10,3]))
b = tf.Variable(tf.zeros(shape=[3]))
output = tf.nn.xw_plus_b(inputs,W,b)
# loss = tf.reduce_mean(tf.square(output-gt))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt,logits=output))
lr = 0.001
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(loss)

bsz = 5
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    num_samples = int(len(x) / bsz)
    for epoch in range(50):
        total_loss = 0.0
        for i in range(num_samples):
            start = i * bsz
            end = min((i+1)*bsz, x.shape[0])
            data = x[start:end]
            gt_input = y[start:end]
            _, tf_loss = sess.run([train_op,loss],feed_dict={inputs:data,gt:gt_input})
            total_loss += tf_loss
        print (epoch , ": " , total_loss)
