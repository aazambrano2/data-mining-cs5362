# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:59:30 2021

@author: Aaron
"""
#Evaluation algorithms
import numpy as np
#scikit-learn

#This algorithm is non-deterministic
from sklearn.cluster import KMeans

#silhouette
from sklearn.metrics import silhouette_score

#Agglomerative
from sklearn.cluster import AgglomerativeClustering


#plotting dendogram
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

#DBSCAN Density Based Clustering of Applications with Noise
from sklearn.cluster import DBSCAN

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def randIndex(knowns, result):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for i in range(0, len(result)):
        for j in range(i+1, len(result)):
            if result[i] == result[j]:
                #positive
                if knowns[i] == knowns[j]:
                    #true positive
                    tp = tp + 1
                else:
                    #false positive
                    fp = fp + 1
            else:
                #Negative
                if knowns[i] == knowns[j]:
                    #false negative
                    fn = fn + 1
                else:
                    #true negative
                    tn = tn + 1
                    
    rand = (tp + tn) / (tp + tn + fp + fn)
    
    return rand

def purity(assignment, known):
    #Confusion matrix
    #holds the cluster levels
    #convert to list so we can use indicies
    assignment_labels = set(assignment)
    assignment_labels = list(assignment_labels)
    
    k_labels = set(known)
    k_labels = list(k_labels)
    
    maxOverlaps = []
    
    #looking at overlapped ID's
    #assignment
    for cID  in assignment_labels:
        #return a list where it gives the incidies of the cluster levels
        #List comprehension
        indiciesOfCID = [ii for ii in range(0, len(assignment)) if assignment[ii] == cID]
        
        overlap = []
        
        #known
        for ckID in k_labels:
             indiciesOfCkID = [ii for ii in range(0, len(known)) if known[ii] == ckID]
             #common indicies with cID and ckID
             overlap.append(len(set(indiciesOfCID).intersection(indiciesOfCkID)))
             
        #max() + max() + max()     
        maxOverlaps.append(max(overlap))
    
    purity = sum(maxOverlaps) / len(assignment)
             
    return purity

#0-1 random numbers
d1 = np.random.rand(10,2)
#2-3 range
d2 = 2+np.random.rand(10,2)
d3 = 5 + np.random.rand(10,2)
d4 = np.random.rand(10,2)

data = np.append(d1, d2, axis=0)
data = np.append(data, d3, axis=0)
data = np.append(data, d4, axis=0)

#first 10 rows and last 10 rows should be in the same cluster because similar values

#cannot go over n rows for clusters
km = KMeans(n_clusters=3).fit(data)

#Prints the assignments of each row where they belong to which cluster
#Cluster assignment vector
print("K-Means assignments for data: ")
print(km.labels_)

#Centroids Each centroid will be 2D because the coords are x,y in this data set

print("Centroids: ")

print(km.cluster_centers_)

#More clusters means a smaller sum of squared distances
print("Sum of squared distances of their closest center: ")
print(km.inertia_)

#list that contains 15 elements with each containing the value of 1
#these are our GT (the knowns)
knowns1 = [1]*15
knowns2 = [2]*10
knowns3 = [3] *15

knowns = knowns1+ knowns2 + knowns3

print("The rand index is: ")
ri = randIndex(knowns, km.labels_)
print(ri)


#p = purity([2,1,1,3,2,2,2,3,2,1,1,3,3,2], [1,2,3,3,1,1,1,3,2,2,1,3,1,2])

p = purity(km.labels_, knowns)
print("Purity: ", p)

#Average silhouette cofficient (Unsupervised)

asc = silhouette_score(data, labels = km.labels_)

print("Average Silhouette Coefficient: ", asc)

#asc < 0 --> bad clustering
#asc = 0 --> no structure
#asc > 0 --> some level of stuctures (We consider the clusters to be more structured)

#Hiearachal Clustering Algorithm----Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3).fit(data)

print("Agglomerative Cluserting lables: ")
print(agg.labels_)

'''
#rand index with kmeans result and agglomerative result
print("Rand index between kmeans and agg: ", randIndex(km.labels_, agg.labels_))
'''

asc_agg = silhouette_score(data, labels = agg.labels_)
print("Average Silhouette Coefficient(agg): ", asc_agg)

#Draws the dandogram provided the model

aggH = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data)

plot_dendrogram(aggH, truncate_mode='level', p=3)

plt.xlabel("Points...")
plt.show()

#DBSCAN
clustering = DBSCAN(eps= 1, min_samples= 10).fit(data)
print("DBSCAN result:")
print(clustering.labels_)

'''
you can make the plots show in separate windows
 by changing it in the 
 tools -> preferences -> IPython Console -> Graphics -> Graphics backend
'''






