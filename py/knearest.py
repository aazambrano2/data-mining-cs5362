# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:26:43 2021

@author: Aaron
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors 


#--------------------------Using NUMPY/Crude (brute force) method---------------------------------
def sortkey(item):
    #compare the second element used to compare
    return item[1]

#data_table --> matrix vector -> data point/vector/object(row), k --> int 
#returns a list of indicies ranking the nearest neighboors from smallest to largest
def k_nearest(data_table, vector, k):
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
    return eucl

if __name__ == '__main__':
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
    
    #uses eucledian distance by default distnce and indicies will have k number of elements / values
    #neigherst neighboors expect that target vectros are given as rows of a 2 dimensional array [[3]]
    #use also given.reshape(1,-1)
    #in other words it expects a 2d array
    
    
    distance , indicies = knbr.kneighbors([given])
    print("Using Sk learn: ")
    print("Indicies: ", indicies)
    print("distances: ", distance)
    
    
    
    
