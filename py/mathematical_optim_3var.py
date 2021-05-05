# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:20:22 2021

@author: Aaron
"""
#f(x0, x1) = (x0 + 2x1 - 7)^2 + (2x0 + x1-5)^2)
from scipy import optimize

def myObj(x):
    y = (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1]-5)**2
    return y

x0 = [0.1,5.4] 
xopt = optimize.fmin(myObj, x0)

print("Where the minimum was found: ", xopt)

#what is the minimum value and what point will you recieve the minimum?

#if there are no other local minimum, then you will recieve the same minimum reagardless of the vector

#0 and at (0.9999, 3.0000)