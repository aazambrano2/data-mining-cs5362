# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np



'''
#We cannot do vector operations with just python lists.  
Numpy is useful to convert these to make the operations possible.

Basically Numpy gives you many Matrix Algebra operations.
'''
def eucledian_dist(data):
    #** squared
        
    #Eucledian distance between row 0 and row 2
    
    index1 = 0
    index2 = 2
    
    #Every dimension there is a subtraction
    
    diff = data[index1] - data[index2]
    print('diff', diff)
    
    #squaring of the differences
    sqdiff = diff**2
    print('sqdiff: ',sqdiff)
    
    #now the summation of the square difference
    
    sum_sqdiff = sum(sqdiff)
    
    print('sum of square difference:',sum_sqdiff)
    
    #now the big squareroot
    eucledian_distance = np.sqrt(sum_sqdiff)
    
    print("Eulcidiean distance: ", eucledian_distance)
    
    #short cut version
    eucledian = np.sqrt(sum((data[index1] - data[index2])**2))
    print('Shortcut answer:',eucledian)

#manhatten distance
def manhatten(data):
    #Manhatten uses the measure around the measurement. 
    #Its like trying to walk around a bulding and measure it
    
    index1 = 0
    index2 = 2
    manh = sum(abs(data[index1] - data[index2]))
    print("Manhattan:", manh)
    
    
def jaccard(data):
    
    index1 = 0
    index2 = 2
    
    #Jaccard Simiilarity
    minimums = []
    maximums = []
    
    #Note: Can do it in one for loop, this is for simplicity sake
    for i in range(0, len(data[index1])):
        
        #[9] --> [9,3] --> [9,3,3] --> [9,3,3,4]
        minimums.append(min(data[index1][i], data[index2][i]))
    print(minimums)
    
    for i in range(0, len(data[index1])):
        maximums.append(max(data[index1][i], data[index2][i]))
    print(maximums)
    
    J = sum(minimums) / sum(maximums)
    
    print("Jaccard similarity: ", J)

#Cosine Similarity
def cosine(data):
    
    index1 = 0
    index2 = 2
    
    #dot product
    
    numerator = np.dot(data[index1], data[index2])
    
    print(numerator)
    
    #L2 norms operator is applied in every cell of that vector
    print (np.sqrt(sum(data[index1]**2)))
    print (np.sqrt(sum(data[index2]**2)))
    
    denominator = np.sqrt(sum(data[index1]**2)) * np.sqrt(sum(data[index2]**2))
    
    cosine = numerator / denominator
    
    print("Cosine similarity: ",cosine)
    
    #cosine of 0  = 1 (Unit circle)

#tanimoto similarity   
#(A.B) / (||A||^2 + ||B||^2 - A.B)
def tanimoto(data):
    index1 = 0
    index2 = 2
    
    #dot product
    numerator = np.dot(data[index1], data[index2])
    
    denominator = sum(data[index1]**2) + sum(data[index2]**2) - numerator
    
    tani = numerator / denominator
    
    print("Tanimoto similarity: ", tani)
    
    
    
    
    

if __name__ == "__main__":
    v1 = np.array([5,4,7])
    v2 = np.array([10,15,2])
    data = np.array([[10,3,3,5],
                    [5,4,5,3],
                    [9,4,3,4],  #data was changed here [9, 4, 3, 4]
                    [8,6,2,6],
                    [20,15,10,20]])
    
    eucledian_dist(data)
    manhatten(data)
    jaccard(data)
    cosine(data)
    tanimoto(data)
    

    #print(v1-v2)
    #print(v1**v2)
                
