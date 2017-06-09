#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:26:13 2017

@author: mike
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# Define gradient descent method
def grad_descent(W, grad, alpha):    
    W = np.subtract(W,np.dot(alpha,grad))
    
    return W

# Define foreward prop method
def foreward_prop(x, W):
    a1 = np.dot(W,x)
    y1 = 1/(1+np.exp(-a1))
    a2 = np.multiply(W,y1)
    
    return a2, y1, a1

# Define backprop
def backprop(x, a2, y1, a1, W):
    sens = np.reshape(np.subtract(a2,x),(2,1))
    first = np.multiply(y1,np.reshape(sens,(1,2)))
    
    gdot = y1*(1-y1)
    second = np.dot(np.multiply(gdot,np.dot(W,sens)),np.reshape(x,(1,2)))
    
    return np.add(first,second)

# Define loss function
def reconstruct_error(X, W):
    error = 0
    N_train = len(X[0])
    for i in range(N_train):
        x = X_train[:,i]
        error += np.linalg.norm(x,np.dot(W,x))
        
    return error/N_train

# Define the radius
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
plt.savefig('task2.png')
plt.show()

# Compute the instantaneous gradient
W = np.subtract(np.dot(2,np.random.random_sample((1,2))),1)
x = X_train[:,0]
a2, y1, a1 = foreward_prop(x,W)
inst_grad = backprop(x, a2, y1, a1, W)
print(inst_grad)

# Step size
eps = .000001

# Compute the approximate gradient
approx_grad = []
for i in range(len(W[0])):
    W_plus = np.copy(W)
    W_plus[0,i] += eps
    
    W_minus = np.copy(W)
    W_minus[0,i] -= eps
    
    plus, _, _ = foreward_prop(x,W_plus)
    minus, _, _ = foreward_prop(x,W_minus)
    
    error = np.linalg.norm(np.subtract(x,plus[0])) - np.linalg.norm(np.subtract(x,minus[0]))
    
    approx_grad.append(error/(2*eps))
print(approx_grad)   
##############################################################################################
print('starting part b')

# Define hyperparamters
alpha = .0000001
max_epoch = 100
runs = 5

# Store things
stored_W = []
stored_error = []

# For the number of different weight initializations
for j in range(runs):
# Intialize weights
    W = np.subtract(np.dot(2,np.random.random_sample((1,2))),1)
    error = np.empty(1)
# Train for the number of specified epochs
    for e in range(max_epoch):
        grad_prop = np.zeros((1,2))
# For every training data point
        for i in range(N_train):
            x = X_train[:,i]
        
# Run the foreward prop
            a2, y1, a1 = foreward_prop(x,W)
            
# Backpropogate the error
            inst_grad = backprop(x, a2, y1, a1, W)
            grad_prop = np.concatenate((grad_prop,inst_grad),axis=0)

# Form the proper gradient    
        grad_prop = np.array((np.sum(grad_prop[:,0]),np.sum(grad_prop[:,1])))

# Update Weights
        W = grad_descent(W, grad_prop, alpha)

# Store things
        error = np.append(error,reconstruct_error(X_train, W), axis=0)
    stored_W.append(W)
    stored_error.append(error)

# Find the best weight and also get rid of any errors that are inf so the plot looks nicer
best = 1000000
stored_error = np.copy(stored_error)
final_error = np.zeros((1,max_epoch))
for i in range(runs):
    if np.amax(stored_error[i,:]) < 1000:
        final_error = np.append(final_error,np.reshape(stored_error[i,1:],(1,max_epoch)),axis=0)
        if best > np.amin(stored_error[i,:]):
            best = np.amin(stored_error[i,:])
            best_W = np.copy(stored_W[i])
    
# Plot errors
plt.figure()
for i in range(len(final_error)-1):
    plt.plot(range(max_epoch),final_error[i+1,:])
plt.title('Reconstrucion Error')
plt.xlabel('Epoch')
plt.savefig('task2_partb.png')
######################################################################################################
print('starting part c')

# Define hyperparameters
num_reduced = 100
reduced = np.linspace(0,1,num=num_reduced)
reduced_out = np.empty((1,2))
for i in range(num_reduced):
    temp = np.dot(best_W,reduced[i])
    reduced_out = np.append(reduced_out,temp,axis=0)
    
# Plot the attempted reconstructed circle
plt.figure()
plt.scatter(reduced_out[1:,0],reduced_out[1:,1])
plt.title('Altered Hidden Layer Output')
plt.savefig('task2_partc.png')

