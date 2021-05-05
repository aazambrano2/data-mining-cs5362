# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:52:40 2021

@author: Aaron
"""

from scipy import optimize

def myObj(x):
    y = 2.1 + x**2
    return y

xopt = optimize.fmin(myObj, 5.5)

print("Where the minimum was found: ", xopt)