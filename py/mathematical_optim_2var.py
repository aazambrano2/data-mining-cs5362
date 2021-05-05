# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:11:22 2021

@author: Aaron
"""

from scipy import optimize

def myObj(x):
    y = 3.3 + x[0]**2 + x[1]**2
    return y

x0 = [10,5.4] 
xopt = optimize.fmin(myObj, x0)

print("Where the minimum was found: ", xopt)