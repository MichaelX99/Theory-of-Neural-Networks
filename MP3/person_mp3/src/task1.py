#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 20:37:46 2017

@author: mike
"""
# Import important libraries
import numpy as np
import matplotlib.pyplot as plt

# Define Sigmoid activation function
def sigmoid(x):
    out = np.zeros(1)
    N = len(x)
    for i in range(N):
        temp = 1/(1+np.exp(-x[i]))
        out = np.append(out,temp)
    return np.reshape(out[1:],(N,1))

# Define linear activation function
def linear(x):
    return x

# Define foreward propogation function
def foreward(x_tilde, V_tilde, U, W_tilde, y1):
    a1 = np.dot(V_tilde,x_tilde) + np.dot(U,y1)
    y1 = sigmoid(a1)

    y1_tilde = np.reshape(np.append(y1,1),(3,1))

    a2 = np.dot(W_tilde,y1_tilde)[0,0]
    y2 = linear(a2)
    
    return y2, y1, a1

# Define backpropogation function
def backprop(y2, t, y1, W, x, iterations, a1, U):
    y1_last  = np.reshape(y1[:,np.shape(y1)[1]-1],(2,1))
    sens2 = y2 - t
    y1_tilde = np.reshape(np.append(y1_last,1),(1,3))
    
    dW = np.dot(sens2,y1_tilde)
    
    sens1 = np.dot(W, sens2)
    
    dV = np.dot(sens1,np.transpose(x))
    dU = np.dot(sens1, np.reshape(y1_last,(1,2)))
     
    for i in range(iterations+1):
        first = a1[0,i+1] * (1-a1[0,i+1])
        second = a1[1,i+1] * (1-a1[1,i+1])
        g = np.array(((first,0),(0,second)))
        sens1 = np.dot(np.dot(g,np.transpose(U)),sens1)
        dV += np.dot(sens1,np.transpose(x))
        dU += np.dot(sens1, np.transpose(y1_last))
    dy = np.dot(np.transpose(U),sens1)
        
    return dW, dV, dU, dy

# Define error function
def compute_error(y,t):
    temp = y - t
    return temp**2

# Define gradint descent function
def grad_descent(x, alpha, grad):
    return np.subtract(x,np.dot(alpha,grad))

# Define the number of time sequences to generate
N = 10

# Define the number of time sequences per sample
t = 4

# Randomly generate the time sequences
s = np.random.uniform(0,1,N)

# Initialize and define the samples
x = np.zeros((N-t+1,4))
for i in range(N-t+1):
    x[i] = (s[i], s[i+1], s[i+2], s[i+3])

# Initialize the true weights that are used to generate the data
V_True = np.random.uniform(-1/5,1/5,(2,5))
U_True = np.random.uniform(-1/2,1/2,(2,2))
W_True = np.random.uniform(-1/3,1/3,(1,3))
y1 = np.zeros((2,1))

# Initialize the output of the network
out1 = np.zeros(1)

# Run the forward prop on every sample
for i in range(N-t+1):
    x_tilde = np.reshape(np.append(x[i],1),(5,1))
    
    y2,y1,_ = foreward(x_tilde, V_True, U_True, W_True, y1)
    
    out1 = np.append(out1,y2)
out1 = out1[1:]

# Rerun the foreward prop with a different hidden layer initialization
y1 = np.dot(.5,np.ones((2,1)))
out2 = np.zeros(1)
for i in range(N-t+1):
    x_tilde = np.reshape(np.append(x[i],1),(5,1))

    y2, y1, a1 = foreward(x_tilde, V_True, U_True, W_True, y1)
    
    out2 = np.append(out2,y2)
out2 = out2[1:]

# Plot the output of the two different netork initializations
plt.figure()
plt.scatter(range(N-t+1), out1, color='blue')
plt.scatter(range(N-t+1), out2, color='red')
plt.legend(('Zeros', '.5'))
plt.title('Data Creation')
plt.show()
###############################################################################

# Define important hyperparameters for learning
max_epoch = 6000
alpha = .0001
best_error = 10000000000

# Initialize the W matrix in order to learn it from scratch
W_error = np.zeros(1)
W_tilde = np.random.uniform(-1/3,1/3,(1,3))

# Run over the max number of epochs
for e in range(max_epoch):
    temp = 0
    y1 = np.zeros((2,1))
    a1 = np.zeros((2,1))
    
    # Run over every sample
    for i in range(N-t+1):
        x_tilde = np.reshape(np.append(x[i],1),(5,1))
    
        # Run the foreward prop
        y2, y1_temp, a1_temp = foreward(x_tilde, V_True, U_True, W_tilde, np.reshape(y1[:,i],(2,1)))
        y1 = np.append(y1, y1_temp, axis=1)
        a1 = np.append(a1, a1_temp, axis=1)
    
        # Backprop the error in order to get gradients
        dW,_,_,_ = backprop(y2, out1[i], y1, np.reshape(W_tilde[0,:2],(2,1)), x_tilde, i, a1, U_True)
    
        # Update the W matrix via the found gradient
        W_tilde = grad_descent(W_tilde, alpha, dW)
        
        # Compute the MSE for the epoch
        temp_error = compute_error(y2, out1[i])
        temp += temp_error
        
        # Solve for the best W
        if temp_error < best_error:
            best_error = temp_error
            Best_W = np.copy(W_tilde)
        
    W_error = np.append(W_error, temp/(N-t+1))
    
# Plot the MSE change over the number of epochs
plt.figure()
plt.plot(range(max_epoch),np.log10(W_error[1:]))
plt.title('Learning W')
plt.show()
###############################################################################

# Define import hyperparamters
max_epoch = 5000
alpha = .1
best_error = 1000000000

# Initialize an the U matrix in order to learn
U_error = np.zeros(1)
U = np.random.uniform(-1/2,1/2,(2,2))

# Run over every epoch
for e in range(max_epoch):
    temp = 0
    y1 = np.zeros((2,1))
    a1 = np.zeros((2,1))
    
    # Run over every sample
    for i in range(N-t+1):
        x_tilde = np.reshape(np.append(x[i],1),(5,1))
    
        # Run the foreward prop
        y2, y1_temp, a1_temp = foreward(x_tilde, V_True, U, W_True, np.reshape(y1[:,i],(2,1)))
        y1 = np.append(y1, y1_temp, axis=1)
        a1 = np.append(a1, a1_temp, axis=1)
    
        # Backpropogate the error in order to find the gradient
        _,_,dU,_ = backprop(y2, out1[i], y1, np.reshape(W_True[0,:2],(2,1)), x_tilde, i, a1, U)
    
        # Update the U matrix with the found gradient
        U = grad_descent(U, alpha, dU)
        
        # Compute the MSE over the epoch
        temp_error = compute_error(y2, out1[i])
        temp += temp_error
        
        # Solve for the best U
        if temp < best_error:
            best_error = temp
            Best_U = np.copy(U)
        
    U_error = np.append(U_error, temp/(N-t+1))
    
# Plot the MSE over the epochs
plt.figure()
plt.plot(range(max_epoch),np.log10(U_error[1:]))
plt.title('Learning U')
plt.show()
###############################################################################

# Define important hyperparameters
max_epoch = 5000
alpha = .2
best_error = 100000000

# Initialize all the weights in order to learn them
error = np.zeros(1)
V = np.random.uniform(-1/5,1/5,(2,5))
U = np.random.uniform(-1/2,1/2,(2,2))
W = np.random.uniform(-1/3,1/3,(1,3))

# Run over every epoch
for e in range(max_epoch):
    temp = 0
    y1 = np.zeros((2,1))
    a1 = np.zeros((2,1))
    
    # Run over every sample
    for i in range(N-t+1):
        x_tilde = np.reshape(np.append(x[i],1),(5,1))
    
        # Run the foreward prop
        y2, y1_temp, a1_temp = foreward(x_tilde, V, U, W, np.reshape(y1[:,i],(2,1)))
        y1 = np.append(y1, y1_temp, axis=1)
        a1 = np.append(a1, a1_temp, axis=1)
    
        # Backprop the error in order to get the gradients
        dW, dV, dU,_ = backprop(y2, out1[i], y1, np.reshape(W[0,:2],(2,1)), x_tilde, i, a1, U)
        
        # Compute the MSE for the epoch
        temp_error = compute_error(y2, out1[i])
        temp += temp_error
        
        # Update the best found weights
        if temp_error < best_error:
            best_error = temp_error
            W_best = np.copy(W)
            V_best = np.copy(V)
            U_best = np.copy(U)
            
        # Update the weight matrices with the found gradients
        W = grad_descent(W, alpha, dW)
        V = grad_descent(V, alpha, dV)
        U = grad_descent(U, alpha, dU)
        
    error = np.append(error, temp/(N-t+1))

# Plot the error over the epochs
plt.figure()
plt.plot(range(max_epoch),np.log10(error[1:]))
plt.title('Learning Parameters')
plt.show()

# Define the output of the network
out_test = np.zeros(1)
y1 = np.zeros((2,1))

# Foreward prop every sample we have with our learned best weights
for i in range(N-t+1):
    x_tilde = np.reshape(np.append(x[i],1),(5,1))
    
    y2,y1,_ = foreward(x_tilde, V_best, U_best, W_best, y1)
    
    out_test = np.append(out_test,y2)
out_test = out_test[1:]

# Plot the output of our learned network
plt.figure()
plt.scatter(range(N-t+1), out1, color='blue')
plt.scatter(range(N-t+1), out_test, color='red')
plt.title('Complete Learned Model Output')
plt.legend(('Original','Recreated'))
plt.show()
###############################################################################

# Define important hyperparameters
max_epoch = 5000
alpha = .2
best_error = 100000000

# Initialize all the weights in order to learn them
error = np.zeros(1)
V = np.random.uniform(-1/5,1/5,(2,5))
U = np.random.uniform(-1/2,1/2,(2,2))
W = np.random.uniform(-1/3,1/3,(1,3))
y1_init = np.zeros((2,1))

# Run over every epoch
for e in range(max_epoch):
    temp = 0
    y1 = np.copy(y1_init)
    a1 = np.zeros((2,1))
    
    # Run over every sample
    for i in range(N-t+1):
        x_tilde = np.reshape(np.append(x[i],1),(5,1))
    
        # Run the foreward prop
        y2, y1_temp, a1_temp = foreward(x_tilde, V, U, W, np.reshape(y1[:,i],(2,1)))
        y1 = np.append(y1, y1_temp, axis=1)
        a1 = np.append(a1, a1_temp, axis=1)
    
        # Backprop the error in order to get the gradients
        dW, dV, dU, dy = backprop(y2, out1[i], y1, np.reshape(W[0,:2],(2,1)), x_tilde, i, a1, U)
        
        # Compute the MSE for the epoch
        temp_error = compute_error(y2, out1[i])
        temp += temp_error
        
        # Update the best found weights
        if temp_error < best_error:
            best_error = temp_error
            W_best = np.copy(W)
            V_best = np.copy(V)
            U_best = np.copy(U)
            y_best = np.copy(y1_init)
            
        # Update the weight matrices with the found gradients
        W = grad_descent(W, alpha, dW)
        V = grad_descent(V, alpha, dV)
        U = grad_descent(U, alpha, dU)
        y1_init = grad_descent(y1_init, alpha, dy)
        
    error = np.append(error, temp/(N-t+1))

# Plot the error over the epochs
plt.figure()
plt.plot(range(max_epoch),np.log10(error[1:]))
plt.title('Learning Parameters')
plt.show()

# Define the output of the network
out_test = np.zeros(1)
y1 = np.copy(y_best)

# Foreward prop every sample we have with our learned best weights
for i in range(N-t+1):
    x_tilde = np.reshape(np.append(x[i],1),(5,1))
    
    y2,y1,_ = foreward(x_tilde, V_best, U_best, W_best, y1)
    
    out_test = np.append(out_test,y2)
out_test = out_test[1:]

# Plot the output of our learned network
plt.figure()
plt.scatter(range(N-t+1), out1, color='blue')
plt.scatter(range(N-t+1), out_test, color='red')
plt.title('Initialization Included Model Output')
plt.legend(('Original','Recreated'))
plt.show()