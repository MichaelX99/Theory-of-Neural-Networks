#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
PART A
"""

#import libraries
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#import data
mat_contents = sio.loadmat('../docs/data/task1.mat')
t = mat_contents['t']
x = mat_contents['x']

#make data into a list
list_x = []
list_t = []
for i in range(len(x)):
    list_x.append(x[i][0])
    list_t.append(t[i][0])

#split the data into train and validation sets
train_x = list_x[:10]
train_t = list_t[:10]
validation_x = list_x[10:len(list_x)]
validation_t = list_t[10:len(list_t)]

#define the required feature vector transformation
def feat_vec(x,p):
    power = []
    for i in range(p):
        power.append(np.power(x,i))
    return power

#initialize the lists of transformed data that I will use for regression
X2 = []
X4 = []
X8 = []
X12 = []
for i in range(len(train_x)):
    X2.append(feat_vec(train_x[i],2))
    X4.append(feat_vec(train_x[i],4))
    X8.append(feat_vec(train_x[i],8))
    X12.append(feat_vec(train_x[i],12))
    
#find the MAP minimizing model    
w_map2 = np.ndarray.tolist(np.dot(np.linalg.inv(np.dot(np.transpose(X2),X2)),np.dot(np.transpose(X2),train_t)))
w_map4 = np.ndarray.tolist(np.dot(np.linalg.inv(np.dot(np.transpose(X4),X4)),np.dot(np.transpose(X4),train_t)))
w_map8 = np.ndarray.tolist(np.dot(np.linalg.inv(np.dot(np.transpose(X8),X8)),np.dot(np.transpose(X8),train_t)))
w_map12 = np.ndarray.tolist(np.dot(np.linalg.inv(np.dot(np.transpose(X12),X12)),np.dot(np.transpose(X12),train_t)))
 
#make the model predictions
out2 = np.ndarray.tolist(np.dot(X2,w_map2)) 
out4 = np.ndarray.tolist(np.dot(X4,w_map4))
out8 = np.ndarray.tolist(np.dot(X8,w_map8))
out12 = np.ndarray.tolist(np.dot(X12,w_map12))
    
#plot the figure and save it
plt.figure(figsize=(10,5))
plt.scatter(train_x,train_t,marker='x')
plt.scatter(train_x,out2,marker='+')
plt.scatter(train_x,out4,marker='^')
plt.scatter(train_x,out8,marker='1')
plt.scatter(train_x,out12,marker='o')
plt.legend(['Training Data','p=2','p=4','p=8','p=12'])
plt.title('Training Set')
plt.xlabel('x value')
plt.ylabel('model prediction')
plt.savefig('../docs/media/2a')

#########################################################################################################################
"""
PART B
"""
#define the mse for the training and validation sets
mse_train = []
mse_val = []

#define the maximum feature transformation dimension
max_p = 13

#initialize the max dimension
p_val = []

#iterate over every dimension
for i in range(max_p):
    #transform the training set
    X_train = []
    for j in range((len(train_x))):
        X_train.append(feat_vec(train_x[j],i))
        
    #transform the validation set
    X_val = []
    for j in range((len(validation_x))):
        X_val.append(feat_vec(validation_x[j],i))
        
    #solve for the MSE minimizing model parameters
    w_map = np.dot(np.linalg.inv(np.dot(np.transpose(X_train),X_train)),np.dot(np.transpose(X_train),train_t))
    
    #find the training error and append it
    temp = np.ndarray.tolist(np.dot(X_train,w_map))
    sub = np.ndarray.tolist(np.subtract(train_t,temp))
    train_err = np.linalg.norm(sub,2)**2
    mse_train.append(train_err/len(train_t))
    
    #find the validation error and append it
    temp = np.ndarray.tolist(np.dot(X_val,w_map))
    sub = np.ndarray.tolist(np.subtract(validation_t,temp))
    val_err = np.linalg.norm(sub,2)**2
    mse_val.append(val_err/len(validation_t))
    
    #append the dimension
    p_val.append(i)
    
#find the validation error minimizer
mse_ind = np.argmin(mse_val)
    
#plot and save   
plt.figure(figsize=(10,5))
plt.plot(p_val,np.log(mse_val))
plt.plot(p_val,np.log(mse_train))
plt.scatter(p_val[mse_ind],np.log(mse_val[mse_ind]))
plt.title('Effect of feature vector size on error')
plt.xlabel('p value (size of feature vector)')
plt.ylabel('Log MSE')
plt.legend(['Validation','Train'])
plt.savefig('../docs/media/2b')
############################################################################################################################
"""
PART C
"""
#define the lists
mse_mu_train = []
mse_mu_val = []
rr_w_map = []
#define the order polynomial
N = 6

#make the array of mu values that gave the best optimizer
mu = np.ndarray.tolist(np.arange(0,.15,.001))

#iterate over all the mu's that I want to check
for i in range(len(mu)):
    #transform the training data
    X_train = []
    for j in range((len(train_x))):
        X_train.append(feat_vec(train_x[j],N))
        
    #transform the validation data
    X_val = []
    for j in range((len(validation_x))):
        X_val.append(feat_vec(validation_x[j],N))  
    
    #solve for the optimal model parameters and append it
    w_map = np.ndarray.tolist(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(X_train),X_train),np.dot(mu[i],np.eye(N)))),np.dot(np.transpose(X_train),train_t)))
    rr_w_map.append(w_map)
    
    #solve for the training error
    temp = np.ndarray.tolist(np.dot(X_train,w_map))
    sub = np.ndarray.tolist(np.subtract(train_t,temp))
    train_err = np.linalg.norm(sub,2)**2
    mse_mu_train.append(train_err/len(train_t))
    
    #solve for the validation error
    temp = np.ndarray.tolist(np.dot(X_val,w_map))
    sub = np.ndarray.tolist(np.subtract(validation_t,temp))
    val_err = np.linalg.norm(sub,2)**2
    mse_mu_val.append(val_err/len(validation_t))
    
#find and print the optimal model
print('The minimum MSE Val is ' + str(np.amin(mse_mu_val)))
ind = np.argmin(mse_mu_val)
print('Mu* = ' + str(mu[ind]) + ' at index ' + str(ind))
print('w_map* = ' + str(rr_w_map[ind]))
 
#make the plot and save
plt.figure(figsize=(10,5))
plt.plot(mu,np.log(mse_mu_train))
plt.plot(mu,np.log(mse_mu_val))
plt.scatter(mu[ind],np.log(mse_mu_val[ind]))
plt.title('Effect of regularizer on MSE Error')
plt.xlabel('mu value (regularizer)')
plt.ylabel('Log MSE')
plt.legend(['Train','Validation'])
plt.savefig('../docs/media/2c')
