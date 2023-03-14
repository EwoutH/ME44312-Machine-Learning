#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:54:30 2021

@author: batasoy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
from collections import Counter
import time

db = pd.read_csv('modes.csv')

# %% Training and test data split: Splitting the dataset into training, validation, and test data set using 60:20:20 split for train: validation: test.

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report

db.head()

X = db.iloc[:,4:]
Y = db['Class']
#label encoding is done as model accepts only numeric values
# so strings need to be converted into labels
LE = preprocessing.LabelEncoder()
LE.fit(Y)
Y = LE.transform(Y)

#splitting dataset into train, validation and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 1)
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.25,random_state = 1)

# datapoints also need to be scaled into dataset with mean 0 and std dev = 1
X_train_scale = preprocessing.scale(X_train)
X_val_scale = preprocessing.scale(X_val)
X_test_scale = preprocessing.scale(X_test)

#Output the number of data points in training, validation, and test dataset.
print("Datapoints in Training set:",len(X_train))
print("Datapoints in validation set:",len(X_val))
print("Datapoints in Test set:",len(X_test))

# %% -------------  PART A -------------------
#Train NN models to obtain the accuracy on the test data using your training and validation data.
#Logistic regression (that can handle multi class classification) is provided for you. 

#Missing parts that you need to implement to answer the question are indicated below (with "Implement!")


from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

train_logreg = LogisticRegression(random_state=0,max_iter = 300).fit(X_train_scale,Y_train)

# !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
# Train a Neural Network
#train_nn = ...

#train_nn.fit(..,...)

pred_logreg = train_logreg.predict(X_val_scale)
print("For Logistic Regression: ")
print(classification_report(Y_val, pred_logreg))
print ("Accuracy of the above model is: ",accuracy_score(pred_logreg,Y_val))

# !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
#Predict based on the trained Neural Network using the validation data
#NOTE: You should reach a NN which has a better accuracy than the logistic regression, if  not revisit the specification of your NN

#pred_nn = ...
print("For Neural Network: ")
print(classification_report(Y_val, pred_nn))
print ("Accuracy of the above model is: ",accuracy_score(pred_nn,Y_val))


# %% -------------  PART B -------------------
#We can also extract features such as minute, hour and day from timestamp column as it was not used till now and try to improve the above accuracies

#Feature Extraction from Timestamp column
db.head()
db['Start Time'] = db['Start Time'].astype('datetime64[ns]')
db['hour'] = db['Start Time'].dt.hour
db['minute'] = db['Start Time'].dt.minute
db['day'] = db['Start Time'].dt.day
db.head()

#Again training the classifiers
X = db.iloc[:,4:]
Y = db['Class']
#label encoding is done as model accepts only numeric values
# so strings need to be converted into labels
LE = preprocessing.LabelEncoder()
LE.fit(Y)
Y = LE.transform(Y)

#splitting dataset into train, validation and test data
# datapoints also need to be scaled into dataset with mean 0 and std dev = 1
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 1)
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.25,random_state = 1)
X_train_scale = preprocessing.scale(X_train)
X_val_scale = preprocessing.scale(X_val)
X_test_scale = preprocessing.scale(X_test)

train_logreg2 = LogisticRegression(random_state=0,max_iter = 300).fit(X_train_scale,Y_train)

# !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
#Train a NN with more features as explained above 

#train_nn2 =...

#train_nn2.fit(...,...)

pred_logreg2 = train_logreg2.predict(X_val_scale)
print("For Logistic Regression: ")
print(classification_report(Y_val, pred_logreg2))
print ("Accuracy of the above model is: ",accuracy_score(pred_logreg2,Y_val))

# !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
# Predict based on the trained NN using the validation data
#NOTE: Again you should reach a NN which has a better accuracy than the logistic regression, if  not revisit the specification of your NN


#pred_nn2 = ...

print("For Neural Network: ")
print(classification_report(Y_val, pred_nn2))
print ("Accuracy of the above model is: ",accuracy_score(pred_nn2,Y_val))

# %% -------------  PART C -------------------
#Accuracy of the models should increase by using additional features
#Pick the one with the highest accuracy and apply it to the test data. 

# !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
#NOTE you should reach an accuracy of at least 80% so revisit your models if you cannot reach that. 

#final_res = ...
print ("Accuracy of the above Classifier is: ",accuracy_score(final_res,Y_test))