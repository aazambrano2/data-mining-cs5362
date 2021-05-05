# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:26:41 2021

@author: Aaron
"""

from scipy import optimize
from numpy import array

from scipy.spatial import distance
def myObj(x, data):
    #returns a property of the solution
    #you will use the free variables and the data
    
    #take the average of every column of the data to get the center
    
    #data might be slow if data is large for fmin to converge
    
    dSum = 0.0
    
    for i in range(0,len(data)):
        dSum= dSum + distance.sqeuclidean(data[i,],x)
    
    
    # x which the sumation of the distance from all the points
    return dSum

myData = array([[2.0, 4.0],
                [2.0, 3.0],
                [5.0,2.0]])
x0 = [10.5,15.4] 
xopt = optimize.fmin(myObj, x0, args=(myData,))

print("Where the minimum was found: ", xopt)