#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:01:21 2017

@author: mike
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as la
import matplotlib.patches as mpatches

#Define function
def f(A,b,c,x):
    f = .5*np.dot(np.transpose(x),np.dot(np.transpose(A),x))-np.dot(np.transpose(x),b)+c
    return f

#Define the gradient function
def df(A,x,b):
    y1 = .5*np.dot(A,x[:])
    y2 = .5*np.dot(np.transpose(A),x[:])
    return y1+y2-b[:,0]

#define the L2 norm
def norm(x):
    y = math.sqrt(pow(x[0],2) + pow(x[1],2))
    return y

def back(A,b,c,x,g):
    p = g / norm(g)
    m = pow(norm(g),2)
    tau = .5
    c = .5
    alpha = 1
    check = f(A,b,c,x) - f(A,b,c,x-alpha*p)
    while check[0] <= alpha * m/2:
           alpha *= tau
           check = f(A,b,c,x) - f(A,b,c,x-alpha*p)
    return alpha

#################################################################################

#Set the maximum number of iterations    
its = 10000

#Set the initial conditions
x = np.zeros((2,its))
x[0,0] = .25
x[1,0] = .25

#Define the System of Equation's Parameters
A = np.zeros((2,2))
A[0,0] = 1
A[0,1] = 1
A[1,0] = 1
A[1,1] = 8
#Determine the positivity and rank of the A matrix in order to determine if it is positive-definite
l,v = la.eig(A)
r = la.matrix_rank(A)
b = np.zeros((2,1))
b[0,0] = 1
b[1,0] = 1
c = 1
 
#Define the learning rate and error we wish to achieve
alpha = .2
eps = 0.000001

#Intialize the gradient so we enter into the while loop
grad_norm = np.zeros((1,its))
grad_norm[0,0] = 100

#Initialize the counter so we don't run into an infinite loop
i = 0
while grad_norm[0,i] > eps and (i < its-1):
    g = df(A,x[:,i],b)
    x[:,i+1] = x[:,i] - alpha*g
    grad_norm[0,i+1] = norm(g)
    i += 1


#define the values of the isocurves I want
iso = np.arange(.25,1,.2)
num = len(iso)

#discritize theta
theta = np.arange(0,2*math.pi,.01)

#make the B matrix
eigval = np.diag(pow(l,-.5))
B = np.dot(np.transpose(v),eigval)

N = len(theta)
z = np.zeros((num,N,2))
mal_x = np.zeros((num,N,2))

x0 = x[:,i]

mal_x = np.zeros((num,N,2))
for j in range(num):
    for p in range(N):
        z[j,p,0] = math.cos(theta[p])
        z[j,p,1] = math.sin(theta[p])
        mal_x[j,p,:] = iso[j] * np.dot(np.transpose(B),z[j,p,:]) + x0
    
for j in range(num):
    plt.plot(mal_x[j,:,0],mal_x[j,:,1]) 

plt.plot(x[0,0:i],x[1,0:i])
plt.grid(True)
plt.title('GD with isocurves')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('isocurve.png')
plt.show()

#####################################################################

#Redefine our X vector
x_opt = np.zeros((2,its))
x_opt[0,0] = .25
x_opt[1,0] = .25
x_below = np.zeros((2,its))
x_below[0,0] = .25
x_below[1,0] = .25
x_above = np.zeros((2,its))
x_above[0,0] = .25
x_above[1,0] = .25   
       
#Reintialize the gradient so we enter into the while loop
grad_opt = np.zeros((2,its))
grad_opt[0,0] = 100
grad_below = np.zeros((2,its))
grad_below[0,0] = 100             
grad_above = np.zeros((2,its))
grad_above[0,0] = 100
             
#Initialize step lengths
alpha_opt = np.zeros((1,its))
alpha_below = .01
alpha_above = .24  
      
#Reinitialize the counter so we don't run into an infinite loop
i_opt = 0
g = 0
while abs(max(grad_opt[:,i_opt])) > eps and (i_opt < its-1):
    g = df(A,x_opt[:,i_opt],b)
    a = np.dot(g,g)/np.dot(g,np.dot(A,g))
    alpha_opt[0,i_opt] = a
    x_opt[:,i_opt+1] = x_opt[:,i_opt] - a*g
    grad_opt[:,i_opt+1] = g
    i_opt += 1
    
i_below = 0
g = 0
while abs(max(grad_below[:,i_below])) > eps and (i_below < its-1):
    g = df(A,x_below[:,i_below],b)
    x_below[:,i_below+1] = x_below[:,i_below] - alpha_below*g
    grad_below[:,i_below+1] = g
    i_below += 1
    
i_above = 0
g = 0
while abs(max(grad_above[:,i_above])) > eps and (i_above < its-1):
    g = df(A,x_above[:,i_above],b)
    x_above[:,i_above+1] = x_above[:,i_above] - alpha_above*g
    grad_above[:,i_above+1] = g
    i_above += 1


alpha_iter = np.arange(.01,.2,.01)

N = len(alpha_iter)

x_iter = np.zeros((2,its))
x_iter[0,0] = .25
x_iter[1,0] = .25
          
grad_iter = np.zeros((2,its))
grad_iter[0,0] = 100
         
i_iter = np.zeros((N,1))         

for i in range(N):
    g = 0
    x_iter = np.zeros((2,its))
    x_iter[0,0] = .25
    x_iter[1,0] = .25
    grad_iter = np.zeros((2,its))
    grad_iter[0,0] = 100
    while abs(max(grad_iter[:,int(i_iter[i])])) > eps and (int(i_iter[i]) < its-1):
        g = df(A,x_iter[:,int(i_iter[i])],b)
        x_iter[:,int(i_iter[i])+1] = x_iter[:,int(i_iter[i])] - alpha_iter[i]*g
        grad_iter[:,int(i_iter[i])+1] = g
        i_iter[i] += 1

##Print out how many iterations were required to converge and what the minimum values are    
#print(i_opt)
#print(i_below)
#print(i_above)
#print(x_opt[:,i_opt])
#print(x_below[:,i_below])
#print(x_above[:,i_above])

plt.plot(x_opt[0,0:i_opt],x_opt[1,0:i_opt])
plt.plot(x_below[0,0:i_below],x_below[1,0:i_below])
plt.plot(x_above[0,0:i_above],x_above[1,0:i_above])
plt.grid(True)
green_patch = mpatches.Patch(color='green', label='< Optimal')
blue_patch = mpatches.Patch(color='blue', label='Optimal')
red_patch = mpatches.Patch(color='red', label='> Optimal')
plt.legend(handles=[green_patch,blue_patch,red_patch])
plt.title('Optimal Learning Rate vs. both Large and Small Fixed Learning Rates')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('multi_learning_rate.png')
plt.show()

plt.scatter(alpha_iter,i_iter)
plt.title('The effect of Learning Rate on # of Iterations until Convergence')
plt.xlabel('Learning Rate')
plt.ylabel('Iterations until Convergence')
plt.savefig('alpha_iteration.png')
plt.show()

############################################################

#Redefine our X vector
x_back = np.zeros((2,its))
x_back[0,0] = .25
x_back[1,0] = .25
     
#Reintialize the gradient so we enter into the while loop
grad_back = np.zeros((2,its))
grad_back[0,0] = 100
        
#Initialize the optimal step length
alpha_back = np.zeros((1,its))

#Reinitialize the counter so we don't run into an infinite loop
i_back = 0
g = 0
while norm(grad_back[:,i_back]) > eps and (i_back < its-1):
    g = df(A,x_back[:,i_back],b)
    a = back(A,b,c,x_back[:,i_back],g)
    alpha_back[0,i_back] = a
    x_back[:,i_back+1] = x_back[:,i_back] - a*g/norm(g)
    grad_back[:,i_back+1] = g
    i_back += 1
    
##Print out how many iterations were required to converge and what the minimum values are    
#print(i_back)
#print(x_back[:,i_back])

plt.plot(x_back[0,0:i_back],x_back[1,0:i_back])
plt.title('Backtracking')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.savefig('backtracking.png')
plt.show()