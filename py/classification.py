# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:28:15 2021

@author: Aaron
"""

#Training data for classification
#Training Labels has each rows of training data with class levels. For example row 1 on Training data has class level 1
#Test data has 30 rows, 4 columns.  Used for deciding what class level belongs
#Test Labels --> answers to what class levels it belongs to.

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

#delimeter tab seperated
#we dont have header in file
data_frame = pd.read_csv("TrainingData.txt", delimiter='\t', header=None)

#2d np array
trainingData = data_frame.values

data_frame = pd.read_csv("TrainingLabels.txt", delimiter='\t', header=None)

#labels need to be one dimensional array
#get all values in column 0
trainingLabels= data_frame.values[:, 0]

#do the same for testing text files

data_frame = pd.read_csv("TestData.txt", delimiter='\t', header=None)

testData = data_frame.values

data_frame = pd.read_csv("TestLabels.txt", delimiter='\t', header=None)

testLabels= data_frame.values[:, 0]

nb = GaussianNB()

#probabilites for each word in each classification
model = nb.fit(trainingData, trainingLabels)


predicted_labels = model.predict(testData)
#TBA




###############################################################################
#Logistic Regression Classification - Deterministic

#First Regreassion
df = pd.read_csv("Pecan.csv", delimiter='\t')
#y = b_0 + b1x1 + b2x2 + b3x3 in this case
#Pecan Yield is the dependent variable

#print(df)

data = df.values[:,range(1,5)]

print(data)

y = data[:,3] #pecan yield

x = data[:,range(0,3)]

reg = LinearRegression().fit(x,y)

print('B0: ',reg.intercept_)

print('B1 B2 B3: ',reg.coef_)

#Negative values: The more salinity you have the less production will yield
#Positive values: The more water you give, the more production will yield

#Given these parameters
unseen = [70, 30, 45] , [90, 20.1, 58] , [90, 80.1, 58]

#B0 + B2*water + B2*salinity + B3 *fertilizer
print(reg.predict(unseen))

#Given the real data, how accurate will it predict the original data
print(reg.predict(data[:,[0,1,2]]))
pred= reg.predict(data[:,[0,1,2]])
#difference
print(data[:,[3]][:, 0] - pred)


#Logistic

data = np.array([[10,0],
                 [30,0],
                 [45,0],
                 [57,1],
                 [61,0],
                 [75,1],
                 [71,0],
                 [78,1],
                 [87,1],
                 [99,1]])
y = data[:,1]

x = data[:, [0]]

logreg = LogisticRegression().fit(x,y)

print(logreg.coef_)
print(logreg.intercept_)

sample=np.array([[80],[60],[45]])
#binary classification (only be done with two class levels)
print(logreg.predict(sample))

print(logreg.predict_proba(sample))
#the probability the person if not infected and infected
#[not infected, infected]





