#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:45:37 2017

@author: mike
"""

#https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# Define gradient descent method
def grad_descent(W, grad, alpha):    
    W = np.add(W,np.dot(alpha,grad))
    
    return W

# Define foreward propogation of input 
def foreward_prop(x_tilde, W1_tilde, W2_tilde):
    a1 = np.dot(W1_tilde,x_tilde)
    y1 = np.tanh(a1)
    y1_tilde = np.copy(y1)
    y1_tilde = np.append(y1_tilde,1)
    
    a2 = np.dot(W2_tilde,y1_tilde)
    y2 = 1/(1+np.exp(-a2))
    
    return y2, a2, y1, a1

# Define backpropogation of error
def backprop(x_tilde, t, a1, y1, a2, y2, W2):
    y1_tilde = np.copy(y1)
    y1_tilde = np.append(y1_tilde,1)
    
    gdot = a2*(1-a2)
    sens = gdot*(y2-t)
    grad2 = np.dot(sens,y1_tilde)

    gdot = np.divide(1,np.power(np.cosh(a1),2))
    sens = np.multiply(gdot,np.dot(W2,sens))

    grad1 = np.dot(np.reshape(sens,(len(sens),1)),np.reshape(x_tilde,(1,len(x_tilde))))
    
    return grad2, grad1

# Define function to compute Cross Entropy Loss
def compute_CE(y,t):
    return -(t*np.log(y)+(1-t)*np.log(1-y))
################################################################################################

print('Starting Part A')
# Define number of samples in each set
N_tot = 1000
N_train = 50
N_test = 500
N_val = 450

# randomly initialize points
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

# plot points used in XOR       
plt.figure()
plt.scatter(false_pts[:,0],false_pts[:,1],color='blue')
plt.scatter(true_pts[:,0],true_pts[:,1],color='red')
plt.savefig('./task3_parta.png')
plt.show()

# split data into 
X_train, X_temp, y_train, y_temp = train_test_split(X_tot, y_tot, train_size=N_train)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, train_size=N_test)
#################################################################################################
print('Starting Part B')
# Define hyperparameters
H = 2
max_epoch = 10000
alpha = .001
runs = 10

# error list to plot later
total_error = []

# define the worst error the NN will ever achieve
best_error = 100000

# loop though all the weight initializations that I want
for j in range(runs):
    # Initialize my not augmented weights
    W1 = np.subtract(np.dot(2,np.random.random_sample((H,2))),1)
    W2 = np.subtract(np.dot(np.dot(2,(H)),np.random.random_sample((H,))),H)
    
    weight_error = []
    # loop through every epoch
    for e in range(max_epoch):
        epoch_error = 0
                
        # augment my weights
        W1_tilde = np.concatenate((W1,np.ones((H,1))),axis=1)
        W2_tilde = np.copy(W2)
        W2_tilde = np.append(W2_tilde,1)
        
        # loop through my training set
        for i in range(N_train):
            # set points
            x = X_train[i,:]
            t = y_train[i]
            
            #augment training point
            x_tilde = np.copy(x)
            x_tilde = np.append(x_tilde,1)
                
            # grab my weight
            W1 = np.copy(W1_tilde)[:H,:2]
            W2 = np.copy(W2_tilde)[:H]
            
            # run the foreward prop
            y2, a2, y1, a1 = foreward_prop(x_tilde, W1_tilde, W2_tilde)
            
            # run backprop
            grad2, grad1 = backprop(x_tilde, t, a1, y1, a2, y2, W2)
                
            # update weights
            W1_tilde = grad_descent(W1_tilde, grad1, alpha)
            W2_tilde = grad_descent(W2_tilde, grad2, alpha)
                    
            # compute the loss for this training point
            temp = compute_CE(y2, t)
            
            # update weights if loss is the best yet seen
            if temp < best_error:
                best_error = temp
                W1_star = np.copy(W1_tilde)
                W2_star = np.copy(W2_tilde)
                
            # add error to list in order to plot
            epoch_error += temp
        weight_error.append(epoch_error/N_train)
    total_error.append(weight_error)

total_error = np.copy(total_error)

# plot error
plt.figure()
for i in range(runs):
    plt.plot(range(max_epoch),total_error[i])
plt.title('Error during training')
plt.ylabel('CE Error')
plt.xlabel('Epochs')
plt.savefig('./task3_partb.png')
plt.show()
###########################################################################################
print('Starting Part C')
# run foreward prop on the best weight that was seen in part b on the testing set
targets_c = np.zeros(1)
for i in range(N_test):
    x = X_test[i,:]
    x_tilde = np.copy(x)
    x_tilde = np.append(x_tilde,1)
    y2, _, _, _ = foreward_prop(x_tilde, W1_star, W2_star)
    targets_c = np.append(targets_c,y2)
    
# make a 3d plot of it
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=X_test[:,0],ys=X_test[:,1],zs=targets_c[1:],c=np.random.rand(3,1))
plt.title('Prediction Space')
ax.set_zlabel('Prediction')
plt.savefig('./task3_partc1.png')
plt.show()

# make a 2d plot that should recreate the original XOR plot in part a
good = 0
pos_c = np.empty((1,2))
neg_c = np.empty((1,2))
for i in range(N_test):
    # run checks to see which point should be in what category and also if the point was correctly predicted
    if targets_c[i+1] < .5:
        pos_c = np.append(pos_c,np.reshape(X_test[i],(1,2)),axis=0)
        if X_test[i,0]*X_test[i,1] > 0:
            good += 1
    if targets_c[i+1] >= .5:
        neg_c = np.append(neg_c,np.reshape(X_test[i],(1,2)),axis=0)
        if X_test[i,0]*X_test[i,1] < 0:
            good += 1
            
# plot the predicitions of test set
print('The missclassification rate is '+str((1-(good/N_test))*100))   
plt.figure()
plt.scatter(pos_c[1:,0],pos_c[1:,1],color='blue')
plt.scatter(neg_c[1:,0],neg_c[1:,1],color='red')
plt.title('Testing Set')
plt.savefig('./task3_partc2.png')
plt.show()
#########################################################################################
print('Starting Part D')
# define all the number of hidden nodes that I will test
nodes = [2, 3, 4, 5]
best_weight1 = []
best_weight2 = []

# for every node I want to test
for H in nodes:
    # define the best error I'll see
    best_error = 1000000
    # for every weight initialization
    for j in range(runs):
        # initialize my weights
        W1 = np.subtract(np.dot(2,np.random.random_sample((H,2))),1)
        W2 = np.subtract(np.dot(np.dot(2,(H)),np.random.random_sample((H,))),H)
        
        weight_error = []
        # for every epoch
        for e in range(max_epoch):
            epoch_error = 0
                    
            # augment my weights
            W1_tilde = np.concatenate((W1,np.ones((H,1))),axis=1)
            W2_tilde = np.copy(W2)
            W2_tilde = np.append(W2_tilde,1)
            
            # for every training point I have
            for i in range(N_train):
                # grab my single training point
                x = X_train[i,:]
                t = y_train[i]
                
                # augment my training point
                x_tilde = np.copy(x)
                x_tilde = np.append(x_tilde,1)
                    
                # grab my weights
                W1 = np.copy(W1_tilde)[:H,:2]
                W2 = np.copy(W2_tilde)[:H]
                
                # run the foreward prop on my single training point
                y2, a2, y1, a1 = foreward_prop(x_tilde, W1_tilde, W2_tilde)
                
                # backpropogate the error of my training point through the network
                grad2, grad1 = backprop(x_tilde, t, a1, y1, a2, y2, W2)
                    
                # update my weights
                W1_tilde = grad_descent(W1_tilde, grad1, alpha)
                W2_tilde = grad_descent(W2_tilde, grad2, alpha)
                        
                # compute the cross entropy loss of the predicition
                temp = compute_CE(y2, t)
                
                # if this is the best error I've seen on this number of nodes, grab the weight
                if temp < best_error:
                    best_error = temp
                    W1_star = np.copy(W1_tilde)
                    W2_star = np.copy(W2_tilde)
                    
                # add the errors up
                epoch_error += temp
            weight_error.append(epoch_error/N_train)
    print('Finished node '+str(H))
    best_weight1.append(W1_star)
    best_weight2.append(W2_star)

# test each of the best weights during training on my validation set
best_overal_error = 1000000000
for i in range(len(nodes)):
    valid_error = 0
    for j in range(N_val):
        x_tilde = np.copy(X_val[j])
        x_tilde = np.append(x_tilde,1)
        out, _, _, _ = foreward_prop(x_tilde, np.copy(best_weight1[i]), np.copy(best_weight2[i]))
        valid_error += compute_CE(out, y_val[j])
    # if this number of nodes gives the best validation error then grab it
    if valid_error < best_overal_error:
        best_overal_error = valid_error
        overal_w1 = np.copy(best_weight1[i])
        overal_w2 = np.copy(best_weight2[i])

# run through the test set with the champion model
good = 0
pos_d = np.empty((1,2))
neg_d = np.empty((1,2))
for i in range(N_test):
    x = X_test[i,:]
    x_tilde = np.copy(x)
    x_tilde = np.append(x_tilde,1)
    y2, _, _, _ = foreward_prop(x_tilde, overal_w1, overal_w2)
    # run checks on prediction and it's label
    if y2 < .5:
        pos_d = np.append(pos_d,np.reshape(x,(1,2)),axis=0)
        if x[0]*x[1] > 0:
            good += 1
    if y2 >= .5:
        neg_d = np.append(neg_d,np.reshape(x,(1,2)),axis=0)
        if x[0]*x[1] < 0:
            good += 1
# make the 2d plot of predicitions        
print('The missclassification rate is '+str((1-(good/N_test))*100))
plt.figure()
plt.scatter(pos_d[1:,0],pos_d[1:,1],color='blue')
plt.scatter(neg_d[1:,0],neg_d[1:,1],color='red')
plt.title('Testing Set')
plt.savefig('./task3_partd1.png')
plt.show()

# make the 3d plot of predicitions
targets_d = np.zeros(1)
for i in range(N_test):
    x = X_test[i,:]
    x_tilde = np.copy(x)
    x_tilde = np.append(x_tilde,1)
    y2, _, _, _ = foreward_prop(x_tilde, overal_w1, overal_w2)
    targets_d = np.append(targets_d,y2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=X_test[:,0],ys=X_test[:,1],zs=targets_d[1:],c=np.random.rand(3,1))
plt.title('Prediction Space')
ax.set_zlabel('Prediction')
plt.savefig('./task3_partd2.png')
plt.show()