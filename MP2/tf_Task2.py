#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:06:38 2017

@author: mike
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

R = 5

# Number of training samples
N_train = 100

# Define the circle
phi = 2 * np.pi* np.random.random_sample(N_train)
mew1 = R*np.cos(phi)
mew2 = R*np.sin(phi)

# Pull the random samples from the gaussian
sigma = .25
X_train = np.random.normal((mew1,mew2),sigma)

# Plot the circle
plt.figure()
plt.scatter(X_train[0],X_train[1])
plt.show()
################################################
max_epoch = 100
learning_rate = 0.01


W = tf.Variable(tf.random_normal([1, 2],stddev=.1))

def network(x):
    a = tf.matmul(W,x)
    y1 = tf.nn.sigmoid(a)
    out = tf.matmul(tf.transpose(W),y1)
    
    return out

x = tf.placeholder(tf.float32, (2,None))
y = network(x)

cost = tf.reduce_mean(tf.pow(y-x,2))

init = tf.global_variables_initializer()

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(max_epoch):
         _, c = sess.run([optimizer, cost], feed_dict={x: X_train})
         
         print('Loss = '+str(c)+' during epoch '+str(i))
         
    plotting = sess.run(y, feed_dict={x: X_train})
    
    v = sess.run(W)
    
    plt.figure()
    plt.scatter(plotting[0],plotting[1])
    plt.show()

plt.figure()
plt.scatter(X_train[0],X_train[1])
plt.show()    
    