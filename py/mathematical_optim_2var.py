# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:11:22 2021

@author: Aaron
"""

from scipy import optimize
import math

def myObj(x):
    #q1
    #y = 3*x[0]**4 + 9*x[1]**2 + 2*x[2]**2 + 3
    
    #q3
    #y = 2**x[0]**4 + 3*x[1]**2
    
    #q5
    #y = (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
    y = math.tan(x[1]) + (x[0]*math.sin(x[2]))**2
    return y

x0 = [0,1.5708,3.14159] 
xopt = optimize.fmin(myObj, x0)

print("Where the minimum was found: ", xopt)

#q1
#x0 = [0,1,2] with 17 wrong
#x0 = [0,0,0] with 0 wrong
#x0 = [0,0,0] with 3 correct

#q2 20 is the minimum for y = 5x[0] + 20

#q3 is x = 0, y = 0 for  y = 2**x[0]**4 + 3*x[1]**

#q5 is x = 3, y  = 0.5 for
#y = (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2