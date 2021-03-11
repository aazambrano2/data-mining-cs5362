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
from sklearn.naive_bayes import GaussianNB

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













