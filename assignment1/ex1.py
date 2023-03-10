#!/usr/bin/env python
# coding: utf-8

# ## Exercise 1
# ME44312 Machine Learning for Transport and Multi-Machine Systems

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os


# In[2]:


path = os.getcwd() + '\data\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()


# In[3]:


data.describe()


# In[4]:


data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))


# ### PART A: CALCULATE THE COSTS

# In[5]:


# Implement the missing parts of the cost function method below!
def h(x, theta):
	return theta[0][0] + theta[0][1] * x

def compute_cost(X, y, theta):
    residual = []
    for i in range(len(y)):
        residual.append((h(X[i], theta) - y[i])**2)
    return np.sum(residual) / (2 * X.shape[0])


# In[6]:


# Add a column of ones to the training set so we can use a vectorized solution to computing the cost and gradients.

data.insert(0, 'Ones', 1)
data.head()


# In[7]:


# Initializing the variables, setting X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

#Printing to double-check
X.head()
print(X)
y.head()
print(y)


# In[8]:


# The cost function is expecting numpy arrays. Therefore, convert X and y.
X = np.array(X.values)
y = np.array(y.values)
theta = np.array([[.0, .0]])

# Again, printing to double-check
print(theta)
print(X.shape, theta.shape, y.shape)


# In[9]:


# Computing the cost
cost = compute_cost(X, y, theta)
print(cost)


# ### PART B: GRADIENT DESCENT

# In[10]:


# Implement the missing parts (i.e, calculate  theta[0,j] in the inner loop)
# of the gradient descent method below!

def gradient_descent(X, y, theta, alpha, iters):
    parameters = theta.shape[1]
    cost = np.zeros([iters, 1])

    for i in range(iters):
        error = (X @ theta.T) - y

        for j in range(parameters):
            term = error * X[:, j:j + 1]
            theta[0,j] -= alpha / X.shape[0] * np.sum(term) ## DOES NOT WORK YET

        cost[i, 0] = compute_cost(X, y, theta)

    return theta, cost


# In[11]:


# Initialize the learning rate and the number of iterations
alpha = 0.01
iters = 2000

# Perform gradient descent and print result
g, cost = gradient_descent(X, y, theta, alpha, iters)
print(g)


# In[12]:


# Calculate and print costs
cost2 = compute_cost(X, y, g)
print(cost2)


# In[13]:


# Plot the results
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# In[14]:


# Plot the cost values over the iterations of the gradient descent algorithm
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


# ### PART C: LINEAR REGRESSION WITH MULTIPLE VARIABLES

# In[15]:


# Read the extended data set
path = os.getcwd() + '\data\ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Poplation Density', 'Mobility Hubs', 'Profit'])
print(data2.head())


# In[16]:


# apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    return df_std

# call the z_score function
data2 = z_score(data2)
print(data2.head())


# In[17]:


# Add ones column
data2.insert(0, 'Ones', 1)

# Set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols - 1]
y2 = data2.iloc[:, cols - 1:cols]

# Convert to np arrays and initialize theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.array([[.0, .0, .0]])


# In[18]:


alpha = 0.0025
iters = 5000

# perform linear regression on the data set
g2, cost2 = gradient_descent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
compute_cost(X2, y2, g2)
costsmv = compute_cost(X2, y2, g2)
print(costsmv)


# In[19]:


# Plot the learning porgress (as before)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


# ### PART D: LINEAR REGRESSION WITH SCIKIT-LEARN

# In[20]:


# Instead of implementing these algorithms from scratch, one could also use scikit-learn's linear regression function.
# See skit learn documentation for details

# Implement the linear regression example from PART A with scikit-learn below!
from sklearn import linear_model
model = linear_model.LinearRegression().fit(X, y)


# In[21]:


# Plot the results
f = model.predict(X).flatten()
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(X[:, 1], f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script ex1.ipynb')

