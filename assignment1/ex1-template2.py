# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 19:38:58 2021

@author: fschulte
"""
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
path = os.getcwd() + '\data\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()

data.describe()

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

#%%
#============== PART A: CALCULATE THE COSTS ==================================

# Implement the missing parts of the cost function method below!

def computeCost(X, y, theta):
    
    return np.sum(residual) / (2 * X.shape[0])

# Add a column of ones to the training set so we can use a vectorized solution to computing the cost and gradients.

data.insert(0, 'Ones', 1)

# Initializing the variables, setting X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

#Printing to double-check
X.head()
print (X)
y.head()
print (y)


# The cost function is expecting numpy arrays. Therefore, convert X and y.
X = np.array(X.values)
y = np.array(y.values)
theta = np.array([[.0,.0]])


# Again, printing to double-check
print(theta)
print(X.shape, theta.shape, y.shape)


# Computing the cost 
cost = computeCost(X, y, theta)
print (cost)


#%%
#=========== PART B: GRADIENT DESCENT ========================================= 

# Implement the missing parts (i.e, calculate  theta[0,j] in the inner loop) 
# of the gradient descent method below!

def gradientDescent(X, y, theta, alpha, iters):
    parameters = theta.shape[1]
    cost = np.zeros([iters,1])
    
    for i in range(iters):
        error = (X @ theta.T) - y
        
        for j in range(parameters):
            term = error * X[:,j:j+1]  

                  
        cost[i,0] = computeCost(X, y, theta)
        
    return theta, cost

# Initialize the learning rate and the number of iterations
alpha = 0.01
iters = 1000

# Perform gradient descent and print result 
g, cost = gradientDescent(X, y, theta, alpha, iters)
print (g)

# Calculate and print costs
cost2 = computeCost(X, y, g)
print (cost2)

# Plot the results
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# Plot the cost values over the iterations of the gradient descent algorithm
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

#%%
#============= PART C: LINEAR REGRESSION WITH MULTIPLE VARIABLES ==============

# Read the extended data set 
path = os.getcwd() + '\data\ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Poplation Density', 'Mobility Hubs', 'Profit'])
print (data2.head())

#  Implement the feature scaling below

print (data2.head())

# Add ones column
data2.insert(0, 'Ones', 1)

# Set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# Convert to np arrays and initialize theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.array([[.0,.0,.0]])

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
computeCost(X2, y2, g2)
costsmv = computeCost(X2, y2, g2)
print (costsmv)

# Plot the learning porgress (as before)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

#%%
#========== PART D: LINEAR REGRESSION WITH SCIKIT-LEARN ========================== 

# Instead of implementing these algorithms from scratch, one could also use scikit-learn's linear regression function. 
# See skit learn documentation for details 

# Implement the linear regression example from PART A with scikit-learn below!
from sklearn import linear_model



# Plot the results 
f = model.predict(X).flatten()
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(X[:, 1], f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')














