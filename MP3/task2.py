#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:48:52 2017

@author: mike
"""
# Import important libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Define the linear activation function
def linear(x):
    return x

# Define the foreward propogation function
def foreward(x_tilde, V_tilde, U, W_tilde, y1):
    a1 = np.dot(V_tilde,x_tilde) + np.dot(U,y1)
    y1 = sigmoid(a1)

    y1_tilde = np.reshape(np.append(y1,1),(2,1))

    a2 = np.dot(W_tilde,y1_tilde)
    y2 = linear(a2)
    
    return y2, y1, a1

# Define the backpropogation function
def backprop(y2, t, y1, W, x, iterations, a1, U, tau):
    y1_last  = y1[iterations]
    sens2 = y2 - t
    
    y1_tilde = np.reshape(np.append(y1[iterations],1),(2,1))
    
    dW = np.dot(sens2, np.transpose(y1_tilde))
    
    sens1 = W[0,0] * sens2
    
    dV = np.dot(sens1,np.transpose(x))
    dU = sens1 * y1_last
     
    for i in range(np.min((iterations+1,tau))):
        g = a1[i] * (1-a1[i])
        sens1 = g * U * sens1
        dV += np.multiply(sens1,np.transpose(x))
        dU += sens1 * y1_last
        
    return dW, dV, dU

# Define the error function
def compute_error(y,t):
    temp = y - t
    return temp**2

# Define the gradient descent function
def grad_descent(x, alpha, grad):
    return np.subtract(x,np.dot(alpha,grad))

# Import the data as strings
with open("MackyG17.dat") as fn:
    read_data = fn.readlines()
    
# Convert the data into a list of floats
data = []    
for i in range(len(read_data)):
    data.append(float(read_data[i].replace("\n","")))
    
# Make the list into a numpy array
data = np.array(data)

# Plot the data
plt.figure()
plt.scatter(range(100),data[:100])
plt.title('MacGlass-Key Dataset')
plt.show()
################################################################################

# Define the number of time sequences per sample
l = 17

# Initialize and set the samples and targets
x = np.zeros((50-l-1,l))
d = np.zeros(50-l-1)
for i in range(len(x)):
    for j in range(l):
        x[i,j] = data[i+j]
    d[i] = data[i+l]
    
# Define important hyperparameters
max_epoch = 5000
alpha = .0003

# Initialize the weights for our model
V_tilde = np.random.uniform(-1/l,1/l,(1,l+1))
U = np.random.uniform(-1,1,1)[0]
W_tilde = np.random.uniform(-1/2,1/2,(1,2))

error = np.zeros(1)
best_error = 100000000

# Run over every epoch
for e in range(max_epoch):
    temp = 0
    y1 = np.zeros(1)
    a1 = np.zeros(1)
    
    # Run over every sample
    for i in range(len(x)):
        x_tilde = np.reshape(np.append(x[i],1),(l+1,1))
        
        # Do the foreward prop
        y2, y1_temp, a1_temp = foreward(x_tilde, V_tilde, U, W_tilde, y1[i])
        y1 = np.append(y1,y1_temp)
        a1 = np.append(a1,a1_temp)
        
        # Backpropogate the error to get the gradients
        dW, dV, dU = backprop(y2, d[i], y1, W_tilde, x_tilde, i, a1, U, l)
        
        # Find the MSE error for the epoch
        temp_error = compute_error(y2, d[i])
        temp += temp_error
            
        # Update the best weights
        if temp_error < best_error:
            best_error = temp_error
            best_V = np.copy(V_tilde)
            best_U = np.copy(U)
            best_W = np.copy(W_tilde)
            
        # Update the weights
        W_tilde = grad_descent(W_tilde, alpha, dW)
        V_tilde = grad_descent(V_tilde, alpha, dV)
        U = grad_descent(U, alpha, dU)
        
    error = np.append(error,temp/len(x))
    
error = error[1:]
    
# Plot the error over the epochs
plt.figure()
plt.plot(range(max_epoch),np.log10(error))
plt.scatter(np.argmin(np.log10(error)),np.amin(np.log10(error)))
plt.title('Error learning a first predictor')
plt.show()

# Find the new samples to test
new_x = np.zeros((100-l-1,l))
for i in range(len(new_x)):
    for j in range(l):
        new_x[i,j] = data[i+j]
        
# Run the foreward prop on the new samples
out = np.zeros(1)
y1 = 0
for i in range(len(new_x)):
    x_tilde = np.reshape(np.append(new_x[i],1),(l+1,1))
    
    temp,y1,_ = foreward(x_tilde, best_V, best_U, best_W, y1)
    
    out = np.append(out,temp)
    
out = out[1:]

first_error = 0
for i in range(len(out)):
    first_error += compute_error(out[i],data[i+l+1])

# Plot the model output
plt.figure()
plt.scatter(range(len(out)),out, color='blue')
plt.scatter(range(100-l),data[l:100], color='red')
plt.title('First Predictor Output')
plt.show()
################################################################################

# Initialize new weights
V_tilde = np.random.uniform(-1/l,1/l,(1,l+1))
U = np.random.uniform(-1,1,1)[0]
W_tilde = np.random.uniform(-1/2,1/2,(1,2))

# Run over every epoch
error = np.zeros(1)
best_error = 100000000
for e in range(max_epoch):
    temp = 0
    y1 = np.zeros(1)
    a1 = np.zeros(1)
    
    # Run over every sample
    for i in range(len(x)):
        x_tilde = np.reshape(np.append(x[i],1),(l+1,1))
        
        # Run the foreward prop
        y2, y1_temp, a1_temp = foreward(x_tilde, V_tilde, U, W_tilde, y1[i])
        y1 = np.append(y1,y1_temp)
        a1 = np.append(a1,a1_temp)
        
        # Backpropogate the error in order to get the gradients
        dW, dV, dU = backprop(y2, d[i], y1, W_tilde, x_tilde, i, a1, U, 10)
        
        # Find the MSE over the epoch
        temp_error = compute_error(y2, d[i])
        temp += temp_error
        
        # Update the best found weights
        if temp_error < best_error:
            best_error = temp_error
            best_V = np.copy(V_tilde)
            best_U = np.copy(U)
            best_W = np.copy(W_tilde)
            
        # Update the weights with the found gradients
        W_tilde = grad_descent(W_tilde, alpha, dW)
        V_tilde = grad_descent(V_tilde, alpha, dV)
        U = grad_descent(U, alpha, dU)
        
    error = np.append(error,temp/len(x))
    
error = error[1:]  
    
# Plot the error over the epochs
plt.figure()
plt.plot(range(max_epoch),np.log10(error))
plt.scatter(np.argmin(np.log10(error)),np.amin(np.log10(error)))
plt.title('Error learning the truncated BPTT')
plt.show()

# Find the new samples
new_x = np.zeros((100-l-1,l))
for i in range(len(new_x)):
    for j in range(l):
        new_x[i,j] = data[i+j]
        
# Run the foreward prop over the new samples
out = np.zeros(1)
y1 = 0
for i in range(len(new_x)):
    x_tilde = np.reshape(np.append(new_x[i],1),(l+1,1))
    
    temp,y1,_ = foreward(x_tilde, best_V, best_U, best_W, y1)
    
    out = np.append(out,temp)
    
out = out[1:]

# Plot the model output using the truncated BPTT
plt.figure()
plt.scatter(range(len(out)),out, color='blue')
plt.scatter(range(100-l),data[l:100], color='red')
plt.title('Trucated BPTT Predictor Output')
plt.show()
###############################################################################

# Initialize and define the second prediction samples
x = np.zeros((50-l-2,l))
d = np.zeros(50-l-2)
for i in range(len(x)):
    for j in range(l):
        x[i,j] = data[i+j]
    d[i] = data[i+l+1]

# Initialize wieghts
V_tilde = np.random.uniform(-1/l,1/l,(1,l+1))
U = np.random.uniform(-1,1,1)[0]
W_tilde = np.random.uniform(-1/2,1/2,(1,2))

error = np.zeros(1)
best_error = 100000000

# Run over every epoch
for e in range(max_epoch):
    temp = 0
    y1 = np.zeros(1)
    a1 = np.zeros(1)
    
    # Run over every sample
    for i in range(len(x)):
        x_tilde = np.reshape(np.append(x[i],1),(l+1,1))
        
        # Run the foreward prop
        y2, y1_temp, a1_temp = foreward(x_tilde, V_tilde, U, W_tilde, y1[i])
        y1 = np.append(y1,y1_temp)
        a1 = np.append(a1,a1_temp)
        
        # Backpropgate the error and solve for the gradients
        dW, dV, dU = backprop(y2, d[i], y1, W_tilde, x_tilde, i, a1, U, l)
        
        # Solve the MSE over the epoch
        temp_error = compute_error(y2, d[i])
        temp += temp_error
        
        # Update the best weights
        if temp_error < best_error:
            best_error = temp_error
            best_V = np.copy(V_tilde)
            best_U = np.copy(U)
            best_W = np.copy(W_tilde)
        
        W_tilde = grad_descent(W_tilde, alpha, dW)
        V_tilde = grad_descent(V_tilde, alpha, dV)
        U = grad_descent(U, alpha, dU)
        
    error = np.append(error,temp/len(x))
    
error = error[1:]
    
# Plot the MSE over the epochs
plt.figure()
plt.plot(range(max_epoch),np.log10(error))
plt.scatter(np.argmin(np.log10(error)),np.amin(np.log10(error)))
plt.title('Error learning the second predictor')
plt.show()

# Find the new samples
new_x = np.zeros((100-l-2,l))
for i in range(len(new_x)):
    for j in range(l):
        new_x[i,j] = data[i+j]
        
# Run the foreward prop over every new sample
out = np.zeros(1)
y1 = 0
for i in range(len(new_x)):
    x_tilde = np.reshape(np.append(new_x[i],1),(l+1,1))
    
    temp,y1,_ = foreward(x_tilde, best_V, best_U, best_W, y1)
    
    out = np.append(out,temp)
    
out = out[1:]

second_error = 0
for i in range(len(out)):
    second_error += compute_error(out[i],data[i+l+2])

# Plot the second prediction output
plt.figure()
plt.scatter(range(len(out)),out, color='blue')
plt.scatter(range(100-l),data[l:100], color='red')
plt.title('Second Predictor Output')
plt.show()
################################################################################

first = []
second = []

first_naive_error = []
second_naive_error = []
for i in range(len(new_x)):
    first_predictor = data[i+l+1]
    second_predictor = data[i+l+2]
    
    first.append(first_predictor)
    second.append(second_predictor)
    
    temp1_error = compute_error(first_predictor, data[i+l])
    temp2_error = compute_error(second_predictor, data[i+l])
    
    first_naive_error.append(temp1_error)
    second_naive_error.append(temp2_error)
    
naive_first_error = np.sum(np.divide(first_naive_error,len(first_naive_error)))
print("First Naive Predictor error is " + str(naive_first_error))
print("First Trained Predictor error is " + str(first_error))

naive_second_error = np.sum(np.divide(second_naive_error,len(second_naive_error)))
print("Second Naive Predictor error is " + str(naive_second_error))
print("Second Trained Predictor error is " + str(second_error))    

plt.figure()
plt.scatter(range(len(first)), first, color='blue')
plt.scatter(range(len(second)), second, color='red')
plt.title('Naive Output')
plt.show()

