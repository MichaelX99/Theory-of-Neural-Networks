#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BETWEEN PART D AND E
"""

#import all the relevant libraries needed
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from matplotlib import animation


#import the data
#https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/io.html
mat_contents = sio.loadmat('../docs/data/task1.mat')
t = mat_contents['t']
x = mat_contents['x']


#convert the numpy array into a list and append a one onto it in order to add a weight for a slope
list_x = []
plot_x = []
list_t = []
temp = []
for i in range(len(x)):
    temp.append(float(x[i]))
    plot_x.append(float(x[i]))
    temp.append(1)
    list_x.append(temp)
    list_t.append(t[i][0])
    temp = []

#how many samples we have
N = len(list_x)

#define the set of update equations that make up the sBLR
def param_update(w_map, C, x, t, sigma):
    #lenght of the input
    n = len(x)
    
    #make an identity matrix
    I = np.ndarray.tolist(np.eye(n))
    
    #find the update matrix
    G = np.ndarray.tolist(np.subtract(I,np.divide((np.dot(np.dot(C,x),x)),((sigma**2)+np.dot(np.dot(x,C),x)))))
    
    #find the updated covariance of the posterior of the parameters
    C_new = np.ndarray.tolist(np.dot(G,C))
    
    #find the updated mean of the posterior of the parameters
    w_new = np.ndarray.tolist(np.divide(np.add(np.dot(G,w_map),np.dot(t,np.dot(C,x))),(np.add(sigma**2,np.dot(np.dot(x,C),x)))))
    
    #return them
    return C_new, w_new

#define hyperparameters given to us in the problem statement
sigma = .5
S = 1

#initialize the w_map list with the 0 vector
w_map = []
w_map.append(np.ndarray.tolist(np.zeros((2,))))

#initialize the covariance list with the identity
C = []
C.append(np.ndarray.tolist(np.eye(2)))

#initialize the lower and upper bounds of the predicition interval
low = []
high = []

#define the hyperparameter alpha that governs what size prediction interval I want
alpha = .55

#iterate over all the data
for i in range(N):
    #update the covariance and mean vector and append them onto the list
    C_new, w_new = param_update(w_map[i],C[i],list_x[i],list_t[i],sigma)
    w_map.append(w_new)
    C.append(C_new)
    
    #solve for the upper and lower bounds and append them
    temp = np.sqrt(np.abs(sigma**2 + np.dot(np.transpose(list_x[i]),np.dot(C_new,list_x[i]))))
    l = np.dot(np.transpose(w_new),list_x[i]) - temp*alpha
    h = np.dot(np.transpose(w_new),list_x[i]) + temp*alpha
    low.append(l)
    high.append(h)        
        
#find the MLE parameter mean
opt = np.ndarray.tolist(np.dot(np.linalg.inv(np.dot(np.transpose(list_x),list_x)),np.dot(np.transpose(list_x),list_t)))

#define the frames that I want to save
save_frame = [100, 250, 500, 750] 
###############################################################################################################
"""
PART E
"""
#initialize the index for the last non-singular covariance matrix
last_i = 0  
    
#define the meshgrid that I want my gaussian distribution to be graphed over
x = np.arange(.9, 1.1, .001)
y = np.arange(.9, 1.1, .001)
X, Y = np.meshgrid(x, y)
 
#define characteristics of the plot
fig = plt.figure(figsize=(10,8))
plt.axes(xlim=(.9, 1.1), ylim=(.9, 1.1))
 
#initialize the number of non singular covariance matrices
num_times = 0

# animation function required for the contour plot
#http://stackoverflow.com/questions/16915966/using-matplotlib-animate-to-animate-a-contour-plot-in-python
def cont_animate(i):
    #what iteration am i on
    print(i)
    
    #clear the last graphs content 
    plt.clf()
    
    #access variables that were not passed in
    global last_i
    global X
    global Y
    global C
    global w_map
    global opt
    global Nx
    global Ny
    global num_times
    
    #define the gaussian distribution
    z = bivariate_normal(X, Y, C[i][0][0], C[i][1][1], w_map[i][0], w_map[i][1], C[i][0][1])
    
    #perform the check whether it is a singular matrix
    #http://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
    if (np.isnan(z).any() == False) and (np.count_nonzero(z) == np.prod(np.shape(z))):
        #plot the distribution
        cont = plt.contour(X, Y, z, 10)
        #plot the MAP predicition
        cont = plt.scatter(x=w_map[i][0],y=w_map[i][1],s=100,marker = 'x')
        last_i = i
        num_times += 1
    else:
        z = bivariate_normal(X, Y, C[last_i][0][0], C[last_i][1][1], w_map[last_i][0], w_map[last_i][1], C[last_i][0][1])
        cont = plt.contour(X, Y, z, 10)
        cont = plt.scatter(x=w_map[last_i][0],y=w_map[last_i][1],s=100,marker = 'x')
    
    #plot the true weights
    cont = plt.scatter(x=1,y=1,marker='o')
    
    #plot the MLE predicition
    cont = plt.scatter(x=opt[0],y=opt[1],s=1000,marker='+')
    cont = plt.legend(['MAP','True','MLE'])
    cont = plt.xlabel('w1')
    cont = plt.ylabel('w2')
    cont = plt.title('Parameter Estimation')
    
    #save the frame if its one of the ones that I want
    if i in save_frame:
        fname = '../docs/media/1e_' + str(i)
        plt.savefig(fname)
    
    #return the plot
    return cont

#generate and save the animation defined above
anim = animation.FuncAnimation(fig, cont_animate, frames=N)
anim.save('../docs/media/1e.avi', writer=animation.FFMpegWriter())

######################################################################################################
"""
PART F
"""
#define the figure
fig = plt.figure(figsize=(10,8))

#define the animation to generate the prediction intervals
def PI(i):
    #print what iteration im at
    print(i)
    
    #clear the last graph
    plt.clf()
    
    #plot the low and high PI's and graph the data point
    pi = plt.scatter(plot_x[:i],low[:i],marker='*')
    pi = plt.scatter(plot_x[:i],list_t[:i])
    pi = plt.scatter(plot_x[:i],high[:i],marker='+')
    pi = plt.legend(['Lower Bound','Training Point','Upper Bound'])
    pi = plt.xlabel('x_n')
    pi = plt.ylabel('Output')
    pi = plt.title('Predictive Intervals')
    
    #save the frame
    if i in save_frame:
        fname = '../docs/media/1f_' + str(i)
        plt.savefig(fname)
    
    #return the plot to animate
    return pi

#make the animation
anim = animation.FuncAnimation(fig, PI, frames=N)
anim.save('../docs/media/1f.avi', writer=animation.FFMpegWriter())
