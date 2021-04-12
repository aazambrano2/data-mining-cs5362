# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:06:01 2021

@author: Aaron
"""

import numpy as np
from numpy import array

#Yahoo, Amazon, Microsoft

A = [[1, 1, 0 ],
     [1, 0, 1 ],
     [0, 1, 0]]

arr = array(A,dtype = float)

#Construct M, column stochastic matrix

s = []

for i in range(0, len(A)):
    s.append(np.sum(arr[:,i]))
    
M = arr

for j in range(0, len(A)):
    M[:,j] = M[:,j] / s[j]
    
r = (1+np.zeros([len(M),1])) / len(M)

# Code

rPrev = r

#loop
#--> Next session


