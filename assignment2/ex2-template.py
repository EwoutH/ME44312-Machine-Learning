# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:55:40 2021

@author: fschulte
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
path = os.getcwd() + '\data\ex2data1.txt'
data = pd.read_csv(path, header=None, names=['QC 1', 'QC 2', 'Passed'])
# plot the data
print (data.head())

positive = data[data['Passed'].isin([1])]
negative = data[data['Passed'].isin([0])]

#%%
#============== PART A: PLOT THE DATA ==================================

# make a scatter plot as shown in the example (see matplotlib documentation for details) 
fig, ax = plt.subplots(figsize=(12,8))





#%%
#============== PART B: PLOT THE SIGMOID FUNCTION =======================

# add the missing code to plot the sigmoid function below (again, see matplotlib documentation for details)
def sigmoid(z):
   


nums = np.arange(-10, 10, step=1)

fig, ax = plt.subplots(figsize=(12,8))

#define the ax.plot below

#%%
#============== PART C: CALCULATE THE COSTS =============================

# complete the cost function below
 
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first =
    second = 
    return np.sum(first - second) / (len(X))

# add a 'ones' column for matrix multiplication (just as in exercise 1)
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

X.shape, theta.shape, y.shape

cost(theta, X, y)

cost1 = cost(theta, X, y)

print (cost1)

# compute the gradient (parameter updates) given our training data, labels, and some parameters theta

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad

gradient(theta, X, y)

#%%
#============== PART D: FIND THE OPTIMAL VALUES ============================

# use SciPy's truncated newton (TNC) implementation to find the values
# complete the code below

import scipy.optimize as opt
result = 

cost(result[0], X, y)

cost2 = cost(result[0], X, y)

print (cost2)

#%%
#============== PART E: EVALUATE THE RESULTS ===============================

# Write a function that will output predictions for a dataset X using our learned parameters theta. 
# Complete the predict function below

def predict(theta, X):
    probability = 
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))


#%%
#============== PART F: EVALUATE THE RESULTS ===============================
# complete the code below

from sklearn import linear_model

model.fit(X, y.ravel())

print (model.score(X, y))









