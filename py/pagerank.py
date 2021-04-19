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

#spidercrab
'''
A = [[1, 1, 0 ],
     [1, 0, 0 ],
     [0, 1, 1]]
'''


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
#5 or 6 zeros are good enough
threshold = 0.0000001
rPrev = r

#loop

for i in range(0,1000):
    r = np.matmul(M,rPrev)
    print("Iteration: ",i)
    print("Rank vector: \n",r)
    
    #convergence:
    diff = sum(abs(r - rPrev))[0]
    print(diff)
    if diff < threshold:
        break
    
    rPrev = r
print("Final rank vector: \n", r)


####################################

#with random teleportation


