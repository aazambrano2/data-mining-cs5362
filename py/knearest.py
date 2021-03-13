# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:26:43 2021

@author: Aaron
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors 

import pandas as pd

#reads csv file that it is in the same directory
def read_file():
    #data frame
    df=pd.read_csv('data-Homework1.csv')
    print(df)
    
    #returns some of the data
    print(df.describe)
    
    #to see a particular column.
    print(df['f3'])
    
    #get multiple columns
    
    print("Multiple Columns:")
    clist = ['f1', 'f3', 'f11']
    #clist is a subset of the bigger data frame
    print(df[clist])
    #same equivalent
    print(df[['f1', 'f3', 'f11']])
    
    #Get Rows
    print("One row")
    #iloc provides index of the row you want to retrieve
    print(df.iloc[0])
    
    #Pandas has object type as series which are like dictionaries or onde dimensional array
    # Data frames are like multi dimensional array
    
    #get multiple rows
    print("Multiple rows: ")
    print(df.iloc[[0,3,10,5]])
    
    #get multiple rows AND Multiple columns
    print("Multiple rows: ")
    #iloc[[rows], [columns]]
    #shows the elements that they intersect
    #loc takes only column names
    #you can also do df.loc[[1,2,3,4],['f3','f4','f5']]
    print(df.iloc[[0,3,10,5],[0,2,3]])
    print(df.iloc[[0,3,10],range(2,6)])
    
    #df.values it is a 2d numpy array HELPFUL FOR HW
    my_data = df.values
    print(my_data)
    #print(type(my_data))
    
    #example distance between row 0 and row 10
    
    
    
    


#--------------------------Using NUMPY/Crude (brute force) method---------------------------------
def sortkey(item):
    #compare the second element used to compare
    return item[1]

#data_table --> matrix vector -> data point/vector/object(row), k --> int 
#returns a list of indicies ranking the nearest neighboors from smallest to largest
def k_nearest(data, vector, k):
    #we have to make sure we compute the distane of each of these rows. 
    #in this case len of this list will be len(data_table)
    #make sure you also track the row numbers [r,d]
    
    all_dist = []
    for row in range(0, len(data)):
        dist = euclid(vector,data[row])
        all_dist.append([row,dist])
    
    #Default sorting algorithm (you could use np.argsort) default quicksort (nlogn)
    sorted_result = sorted(all_dist, key = sortkey)
    
    print(sorted_result)
    
    indicies = []
    
    if k <len(data):
        #looping the number of rows
        for i in range(0,k):
            #map to the indexes that are already sorted
            indicies.append(sorted_result[i][0])
    else:
        #k is greater than the number of vectors in the data table
        for i in range(0, len(data)):
            indicies.append(sorted_result[i][0])
    return indicies



def euclid(v1, v2):
    eucl = np.sqrt(sum((v1-v2)**2))
    
    print("Eucledian: ", eucl)
    return eucl

def manhatten(v1,v2):
    #v1 and v2 are indexes
    #Manhatten uses the measure around the measurement. 
    #Its like trying to walk around a bulding and measure it
    
    manh = sum(abs(v1 - v2))
    print("Manhattan:", manh)
    return manh

def jaccard(v1,v2):
    
    #Jaccard Simiilarity
    minimums = []
    maximums = []
    
    #Note: Can do it in one for loop, this is for simplicity sake
    for i in range(0, len(v1)):
        
        #[9] --> [9,3] --> [9,3,3] --> [9,3,3,4]
        minimums.append(min(v1[i], v2[i]))
    #print(minimums)
    
    for i in range(0, len(v1)):
        maximums.append(max(v1[i], v2[i]))
    #print(maximums)
    
    J = sum(minimums) / sum(maximums)
    
    print("Jaccard similarity: ", J)
    return J

#Cosine Similarity
def cosine(v1, v2):
    
    
    #dot product
    
    numerator = np.dot(v1, v2)
    
    print(numerator)
    
    #L2 norms operator is applied in every cell of that vector
    print (np.sqrt(sum(v1**2)))
    print (np.sqrt(sum(v2**2)))
    
    denominator = np.sqrt(sum(v1**2)) * np.sqrt(sum(v2**2))
    
    cosine = numerator / denominator
    
    print("Cosine similarity: ",cosine)
    
    #cosine of 0  = 1 (Unit circle)
    return cosine

#tanimoto similarity   
#(A.B) / (||A||^2 + ||B||^2 - A.B)
def tanimoto(v1,v2):
    
    
    #dot product
    numerator = np.dot(v1, v2)
    
    denominator = sum(v1**2) + sum(v2**2) - numerator
    
    tani = numerator / denominator
    
    print("Tanimoto similarity: ", tani)
    
    return tani
    
if __name__ == '__main__':
    '''
    data = np.array([[10,3,3,5,10],
                     [5,4,5,3,6],
                     [10,4,6,4,9],
                     [8,6,2,6,3],
                     [10,3,3,5,8],         
                     [9,2,1,2,11],
                     [9,3,1,2,11]])
    
    print()
    given = data[1]
    k=3
    k_list = k_nearest(data,given,k)
    print("The indicies of the nearest neighboors are: " , k_list)
    #nearest neighboor of row 1 is itself. (distance is 0) tricky question
    
    
    #------------------------Using sklearn---------------------------------
    #n_neighbors => k algorithm --> data structure .fit Returns an object k nearest neighboors
    knbr = NearestNeighbors(n_neighbors= k, algorithm = 'ball_tree').fit(data)
    #auto , kd_tree, ball tree
    
    #uses eucledian distance by default distnce and indicies will have k number of elements / values
    #neigherst neighboors expect that target vectros are given as rows of a 2 dimensional array [[3]]
    #use also given.reshape(1,-1)
    #in other words it expects a 2d array
    
    
    distance , indicies = knbr.kneighbors([given])
    print("Using Sk learn: ")
    print("Indicies: ", indicies)
    print("distances: ", distance)
    
    data2 = np.array([[10,3,3,5],
                    [5,4,5,3],
                    [9,4,3,4],  #data was changed here [9, 4, 3, 4]
                    [8,6,2,6],
                    [20,15,10,20]])
    
    v1 = data2[0]
    
    v2 = data2[2]
    
    '''
    
    
    '''
    euclid(v1,v2)
    manhatten(v1,v2)
    jaccard(v1,v2)
    cosine(v1,v2)
    tanimoto(v1,v2)
    '''
    
    
    
    '''
    HOMEWORK 1
    '''
    df=pd.read_csv('data-Homework1.csv')
    
    #Question 1:
    '''
    Jaccard similarity between Rows 0 and 1 is 0.7523955174360798.
    What is the Jaccard similarity between Rows 5 and 75? 
    Your answer must be correct up to four digits after the decimal point. 
    No exponent notations are allowed in the answer. 
    '''
    data_set = df.values
    print(data_set)
    
    
    
    jaccard(data_set[5],data_set[75]) #0.7113930622201977
    
    
    #knearest neighboor -> row 65 [[65 62 27 97 86]]
    #row 75 -> [[75 57 52 93 62]]
    given = data_set[75] 
    k=5
    #------------------------Using sklearn---------------------------------
    #n_neighbors => k algorithm --> data structure .fit Returns an object k nearest neighboors
    knbr = NearestNeighbors(n_neighbors= k, algorithm = 'ball_tree').fit(data_set)
    #auto , kd_tree, ball tree
     
    distance , indicies = knbr.kneighbors([given])
    print("Using Sk learn: ")
    print("Indicies: ", indicies)
    print("distances: ", distance)
    
    
    cosine(data_set[5], data_set[100]) #0.9136906342448988
    
    tanimoto(data_set[5], data_set[75])#0.9229486033454978
    
    euclid(data_set[5], data_set[100])#3.554169691976415
    
    
    
    
    
    
    
    
    #print(0.9823746957909885 == 0.9823746957909885)
    
    #read_file()
    
    
    
    
    
    
