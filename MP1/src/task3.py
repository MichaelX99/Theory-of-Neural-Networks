#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
PART A
"""

#import libraries
import numpy as np
import csv
import matplotlib.pyplot as plt

# Import the data
with open('../docs/data/housing.csv') as f:
    reader = csv.reader(f, delimiter=",")
    d = list(reader)

# Convert the data to lists
shape = np.shape(d)
data = []
sample = []
for i in range(shape[0]):
    for j in range(shape[1]):
        sample.append(float(d[i][j]))
    data.append(sample)
    sample  = []

#grab the train chunk
train = np.copy(data[:305])
x_train = np.ndarray.tolist(train[:,:shape[1]-1])
y_train = np.ndarray.tolist(train[:,shape[1]-1])

#grab the validation chunk
validation = np.copy(data[306:406])
x_valid = np.ndarray.tolist(validation[:,:shape[1]-1])
y_valid = np.ndarray.tolist(validation[:,shape[1]-1])

#grab the test chunk
test = np.copy(data[406:shape[0]])
x_test = np.ndarray.tolist(test[:,:shape[1]-1])
y_test = np.ndarray.tolist(test[:,shape[1]-1])

#define hyperparemeters
mu = 0
s = 1
w_0 = np.random.normal(mu,s,shape[1]-1)
 
#initialize lists
mse = []
rr_w_star = []
 
#define the mu list that I want to search over
rr_mu = np.ndarray.tolist(np.arange(0,100,.1))
 
#iterate over every mu
for i in range(len(rr_mu)):
    mu = rr_mu[i]
    
    #find the optimal model parameters and append them
    w_star = np.ndarray.tolist(np.dot(np.linalg.inv(np.add(np.dot(np.transpose(x_train),x_train),np.dot(mu,np.eye(shape[1]-1)))),np.dot(np.transpose(x_train),y_train)))
    rr_w_star.append(w_star)
    
    #solve for the validation error and append it
    temp = np.ndarray.tolist(np.dot(x_valid,w_star))
    sub = np.ndarray.tolist(np.subtract(y_valid,temp))
    valid_err = np.linalg.norm(sub,2)**2
    mse.append(valid_err/len(y_valid))
    
#find the optimal mu
rr_ind = np.argmin(mse)
 
#plot it
plt.figure(figsize=(10,5))
plt.scatter(rr_mu[rr_ind],np.log(mse[rr_ind]))
plt.plot(rr_mu,np.log(mse))
plt.title('Regularization')
plt.xlabel('mu value')
plt.legend(['MSE on validation set'])
plt.ylabel('Log MSE on Validation Set')
plt.savefig('../docs/media/3a')

######################################################################################################
"""
PART B
"""
#define the function that will solve for the median distance of all the training data
def gamma_solver(x_n):
    out = []
    #loop over all of the data twice
    for i in range(len(x_n)):
        for j in range(len(x_n)):
            #find the norm squared and append it
            diff = np.subtract(x_n[j],x_n[i])
            out.append(np.linalg.norm(diff,2)**2)
    
    #return the median norm squared
    return np.median(out)

#define the function that will transform my data
def RBF(x,x_n,gamma):
    out = []
    #loop  over and append the transformed vector
    for i in range(len(x_n)):
        diff = np.subtract(x_n[i],x)
        norm = np.linalg.norm(diff,2)**2
        prod = np.exp(-gamma * norm)
        out.append(prod)
    return out

#find the optimal starting search point for gamma
opt = gamma_solver(x_train)

#define the search window for gamma that gave an actual minimizer
rbf_gamma = np.ndarray.tolist(np.arange(start=5/opt,stop=20/opt,step=.1/opt))

#define lists
rbf_w_map = []
rbf_mse = []

#loop over every gamma that I care about
for i in range(len(rbf_gamma)):
    gamma = rbf_gamma[i]
 
    #transform the training data
    X_train = []
    for j in range(len(x_train)):
        X_train.append(RBF(x_train[j],x_train,gamma))
        
    #transform the validaiton data
    X_val = []
    for j in range(len(x_valid)):
        X_val.append(RBF(x_valid[j],x_train,gamma))
    
    #check if the the transformed data is singular
    inter = np.ndarray.tolist(np.dot(np.transpose(X_train),X_train))
    if np.linalg.cond(inter) == np.inf:
        #it is therefore use the moore penrose pseudo inverse
        w_map = np.ndarray.tolist(np.dot(np.linalg.pinv(inter),np.dot(np.transpose(X_train),y_train)))
    else:
        #its not so use the regular invers
        w_map = np.ndarray.tolist(np.dot(np.linalg.inv(inter),np.dot(np.transpose(X_train),y_train)))
        
    #append the found model parameter to a list
    rbf_w_map.append(w_map)
    
    #find and append the erro
    error = np.ndarray.tolist(np.subtract(y_valid,np.dot(X_val,w_map)))  
    mse_val = (np.linalg.norm(error,2)**2)
    rbf_mse.append(mse_val/len(x_valid))
    
    #what iteration am i on
    print(i)
 
#find where the mse was minimized
rbf_ind = np.argmin(rbf_mse)
 
#plot it
plt.figure(figsize=(10,5))
plt.plot(rbf_gamma,np.log(rbf_mse))
plt.title('RBF ANN Gaussian Kernel')
plt.xlabel('gamma value')
plt.ylabel('Log MSE on Validation Set')
plt.scatter(rbf_gamma[rbf_ind],np.log(rbf_mse[rbf_ind]))
plt.savefig('../docs/media/3b') 

#tell me where
print('min occurs at ' + str(rbf_ind))
############################################################################################################
"""
PART C
"""
#transform the test data with the RBF
X_test = []
for i in range(len(test)):
    X_test.append(RBF(test[i,:shape[1]-1],train[:,:shape[1]-1],rbf_gamma[rbf_ind]))
    
#define the averaging LR model
w_lr = []
for i in range(13):
    w_lr.append(1/13)
 
#solve for the LR output
lr = np.dot(test[:,:shape[1]-1],w_lr)
    
#find the MSE for all 3 champion models
rr_test_mse = np.dot(test[:,shape[1]-1]-np.dot(test[:,:shape[1]-1],rr_w_star[rr_ind]),test[:,shape[1]-1]-np.dot(test[:,:shape[1]-1],rr_w_star[rr_ind]))/len(test)
rbf_test_mse = np.dot(test[:,shape[1]-1] - np.dot(X_test,rbf_w_map[rbf_ind]),test[:,shape[1]-1] - np.dot(X_test,rbf_w_map[rbf_ind]))/len(test)
lr_test_mse = np.dot(np.subtract(test[:,shape[1]-1],lr),np.subtract(test[:,shape[1]-1],lr))/len(test)
 
#print out the error so that I can choose which is the best
print('RR Error is ' + str(rr_test_mse))
print('RBF Error is ' + str(rbf_test_mse))
print('LR Error is ' + str(lr_test_mse))
