#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:05:09 2017

@author: mike
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

#http://stackoverflow.com/questions/41037650/how-to-restore-session-in-tensorflow
#https://matplotlib.org/examples/mplot3d/rotate_axes3d_demo.html

N_tot = 1000
N_train = 50
N_test = 500
N_val = 450

X_tot = 2*np.random.random_sample((N_tot,2))-1
y_tot = np.zeros((N_tot,))

y_tot[X_tot[:,0]*X_tot[:,1]<0] = 1

true_pts = np.empty((1,2))
false_pts = np.empty((1,2))

for i in range(N_tot):
    if y_tot[i] == 0:
        false_pts = np.append(false_pts,np.reshape(X_tot[i],(1,2)),axis=0)
    if y_tot[i] == 1:
        true_pts = np.append(true_pts,np.reshape(X_tot[i],(1,2)),axis=0)
        
plt.figure()
plt.scatter(false_pts[:,0],false_pts[:,1],color='blue')
plt.scatter(true_pts[:,0],true_pts[:,1],color='red')

X_train, X_temp, y_train, y_temp = train_test_split(X_tot, y_tot, train_size=N_train)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, train_size=N_test)
####################################################################################
max_epoch = 15000
learning_rate = 0.001

def MLP(x):
    a1 = tf.add(tf.matmul(W1,x),b1)
    y1 = tf.tanh(a1)
    a2 = tf.add(tf.matmul(W2,y1),b2)
    y2 = tf.nn.sigmoid(a2)
    
    return y2

runs = 10
H = 2

best_error = 10000000
total_error = []
for e in range(runs):
    tf.reset_default_graph()
    
    W1 = tf.Variable(tf.random_uniform([H,2],minval=-1,maxval=1), name='W1')
    W2 = tf.Variable(tf.random_uniform([1,H],minval=-1,maxval=1), name='W2')

    b1 = tf.Variable(tf.random_uniform([H,1],minval=-1,maxval=1), name='b1')
    b2 = tf.Variable(tf.random_uniform([1,1],minval=-1,maxval=1), name='b2')
    
    tf.add_to_collection('vars', W1)
    tf.add_to_collection('vars', W2)
    tf.add_to_collection('vars', b1)
    tf.add_to_collection('vars', b2)
    
    x = tf.placeholder(tf.float32, (2,None))
    #x = tf.placeholder(tf.float32, (2,1))
    target = tf.placeholder(tf.float32, None)

    model = MLP(x)

    cost = tf.reduce_sum(-tf.add(tf.multiply(target,tf.log(model)),tf.multiply(tf.subtract(1.0,target),tf.log(tf.subtract(1.0,model)))))

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    init = tf.global_variables_initializer()

    tf_saver = tf.train.Saver({'W1':W1, 'W2':W2, 'b1':b1, 'b2':b2})

    model_checkpoint = "tf_task3"

    with tf.Session() as sess:
        sess.run(init)
        for i in range(max_epoch):
            train_error = 0
            #for j in range(N_train):
            #    _, c = sess.run([optimizer,cost],feed_dict={x: np.reshape(X_train[j],(2,1)), target: y_train[j]})
            #    train_error += c
            #    if c < best_error:
            #        best_error = c
            #        tf_saver.save(sess, model_checkpoint)
            _, c = sess.run([optimizer,cost],feed_dict={x: np.reshape(X_train,(2,len(X_train))), target: y_train})
            if np.amax(c) < best_error:
                    best_error = np.amax(c)
                    tf_saver.save(sess, model_checkpoint)
            if i % 1000 == 0:
                print('Finished epoch '+str(i))
        #train_error /= N_train
        train_error = np.sum(c)/N_train
        total_error.append(train_error)
    
    print('Finished weight '+str(e)+' with error '+str(train_error))
        
plot = np.empty((runs,max_epoch))
for i in range(runs):
    plot[i] = np.copy(total_error[i])
plt.figure()
for i in range(runs):
    plt.plot(range(max_epoch),plot[i,:],c=np.random.rand(3,1))
plt.show()
########################################################################################
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(model_checkpoint+'.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    all_vars = tf.get_collection('vars')    
    sess.run(init)
    
    W1_star = sess.run(W1)
    W2_star = sess.run(W2)
    b1_star = sess.run(b1)
    b2_star = sess.run(b2)
    
    
    #plot = []
    #for i in  range(N_train):
    #    temp = sess.run(model,feed_dict={x: np.reshape(X_train[i],(2,1))})
    #    plot.append(temp)
    #plot = np.array(plot)
    plot = sess.run(model,feed_dict={x: np.reshape(X_train,(2,len(X_train)))})
    plot = np.reshape(plot,(N_train,))
    
    pos = np.empty((1,2))
    ns = np.empty((1,2))
    neg = np.empty((1,2))
    for i in range(N_train):
        if plot[i] < .3:
            pos = np.append(pos,np.reshape(X_train[i],(1,2)),axis=0)
        if (plot[i] >= .3) & (plot[i] <= .7):
            ns = np.append(ns,np.reshape(X_train[i],(1,2)),axis=0)
        if plot[i] > .7:
            neg = np.append(neg,np.reshape(X_train[i],(1,2)),axis=0)
    plt.figure()
    plt.scatter(pos[1:,0],pos[1:,1],color='blue')
    plt.scatter(neg[1:,0],neg[1:,1],color='red')
    plt.scatter(ns[1:,0],ns[1:,1],color='black')
    plt.show()
    
    
    #plot = []
    #c = 0
    #for i in  range(N_test):
    #    temp_plot, temp_c = sess.run([model,cost],feed_dict={x: np.reshape(X_test[i],(2,1)), target: y_test[i]})
    #    plot.append(temp_plot)
    #    c += temp_c
    #plot = np.array(plot)
    #c /= N_test
    
    plot, c = sess.run([model, cost],feed_dict={x: np.reshape(X_test,(2,len(X_test))), target: y_test})
    plot = np.reshape(plot,(N_test,))
    
    c = np.sum(c)/N_test
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=X_test[:,0],ys=X_test[:,1],zs=plot,c=np.random.rand(3,1))
    plt.show()
    
    pos = np.empty((1,2))
    ns = np.empty((1,2))
    neg = np.empty((1,2))
    for i in range(N_test):
        if plot[i] < .3:
            pos = np.append(pos,np.reshape(X_test[i],(1,2)),axis=0)
        if (plot[i] >= .3) & (plot[i] <= .7):
            ns = np.append(ns,np.reshape(X_test[i],(1,2)),axis=0)
        if plot[i] > .7:
            neg = np.append(neg,np.reshape(X_test[i],(1,2)),axis=0)
    plt.figure()
    plt.scatter(pos[1:,0],pos[1:,1],color='blue')
    plt.scatter(neg[1:,0],neg[1:,1],color='red')
    plt.scatter(ns[1:,0],ns[1:,1],color='black')
    plt.show()
    
    print('Testing error = '+str(c))
########################################################################################
#H = [2, 3, 4, 5]
#testing = []
#max_epoch = 1000
#for j in H:
#    best_error = 10000000
#    for e in range(10):
#        tf.reset_default_graph()
#        W1 = tf.Variable(tf.random_normal([j,2],stddev=.1), name='W1')
#        W2 = tf.Variable(tf.random_normal([1,j],stddev=.1), name='W2')
#        
#        b1 = tf.Variable(tf.random_normal([j,1],stddev=.1), name='b1')
#        b2 = tf.Variable(tf.random_normal([1,1],stddev=.1), name='b2')
#        
#        tf.add_to_collection('vars', W1)
#        tf.add_to_collection('vars', W2)
#        tf.add_to_collection('vars', b1)
#        tf.add_to_collection('vars', b2)
#        
#        x = tf.placeholder(tf.float32, (None,2))
#        target = tf.placeholder(tf.float32, (None))
# 
#        model = MLP(x)
# 
#        cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=model))
# 
#        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 
#        init = tf.global_variables_initializer()
#        
#        tf_saver = tf.train.Saver({'W1':W1, 'W2':W2, 'b1':b1, 'b2':b2})
#        
#        train_error = []
#        with tf.Session() as sess:
#            sess.run(init)
#            for i in range(max_epoch):
#                _, c = sess.run([optimizer,cost],feed_dict={x: X_train, target: y_train})
#            if c < best_error:
#                best_error = c
#                tf_saver.save(sess, model_checkpoint)
#                
#    with tf.Session() as sess:        
#        new_saver = tf.train.import_meta_graph(model_checkpoint+'.meta')
#        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
#        all_vars = tf.get_collection('vars')
#        sess.run(init)
#        plot = sess.run(model,feed_dict={x: X_test})
#        plot = np.reshape(plot,(N_test,))
#    
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(xs=X_test[:,0],ys=X_test[:,1],zs=plot,c=np.random.rand(3,1))
#        ax.view_init(0, 45)
#        plt.show()
#         
#        pos = np.empty((1,2))
#        ns = np.empty((1,2))
#        neg = np.empty((1,2))
#        for i in range(N_test):
#            if plot[i] < .3:
#                pos = np.append(pos,np.reshape(X_test[i],(1,2)),axis=0)
#            if (plot[i] >= .3) & (plot[i] <= .7):
#                ns = np.append(ns,np.reshape(X_test[i],(1,2)),axis=0)
#            if plot[i] > .7:
#                neg = np.append(neg,np.reshape(X_test[i],(1,2)),axis=0)
#        plt.figure()
#        plt.scatter(pos[1:,0],pos[1:,1],color='blue')
#        plt.scatter(neg[1:,0],neg[1:,1],color='red')
#        plt.scatter(ns[1:,0],ns[1:,1],color='black')
#        plt.show()
#        print(str(j)+' # of hidden nodes')