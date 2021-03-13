# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:24:56 2021

@author: Aaron
"""
#pandas 
import pandas as pd
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


if __name__ == "__main__":
    
    #print(randIndex([3,1,3,2,1],[1,3,3,2,1]))
    
    #2 cluseters for the practice data
    #6 clusters according to assignment_labels.csv file
    #Test data confirmed to be correct
    '''
    practice_data = pd.read_csv('lab2_practice_data.csv', header = None)
    
    test_data = practice_data.values
    
    #print(test_data)
    
    practice_labels = pd.read_csv('lab2_practice_labels.csv', header=None)
    
    test_labels = practice_labels.values[:,0] #knows for practice data
    
    km = KMeans(n_clusters=2).fit(test_data)
    agg = AgglomerativeClustering(n_clusters=2).fit(test_data)
    clustering = DBSCAN(eps= 0.3, min_samples= 10).fit(test_data)
    
    #print(clustering.labels_)
    
    #---Rand Index
    print("Rand Index ")
    
    #Kmeans
    print('Kmeans Rand Index: ',randIndex(test_labels, km.labels_))
    
    #Agg
    print('Agglomerative Rand Index: ',randIndex(test_labels, agg.labels_))
    
    #DBSCAN

    print('DBSCAN Rand Index: ',randIndex(test_labels, clustering.labels_))
    
    print("-----------------------------")
    
    #---Purity
    print("Purity")
    
    #Kmeans
    print('Kmeans Purity: ',purity(km.labels_, test_labels))
    
    #Agg
    print('Agglomerative Purity: ',purity(agg.labels_, test_labels))
    
    #DBSCAN
    print('Kmeans Purity: ',purity(clustering.labels_, test_labels))
    print("-----------------------------")
    
    #---Silhouette
    print("Silhouette")
    
    #Kmeans
    print('Kmeans Silhouette: ',silhouette_score(test_data, labels = km.labels_) )
    
    #Agg
    print('Agglomerative Silhouette: ',silhouette_score(test_data, labels = agg.labels_))
    
    #DBSCAN (last two digits different)
    print('DBSCAN Silhouette: ', silhouette_score(test_data, labels = clustering.labels_))
    '''
    
    print('\nASSIGNMENT 2\n')
    
    #--------------------------------------------------------------------------
    assignment_data = pd.read_csv('assignment2_data.csv', header = None)
    
    test_data = assignment_data.values
    
    #print(test_data)
    
    assignment_labels = pd.read_csv('assignment2_labels.csv', header=None)
    
    test_labels = assignment_labels.values[:,0] #knows for practice data
    
    km = KMeans(n_clusters=6).fit(test_data)
    '''
    maximum = 0
    ri = 0
    for i in range(100):
        km = KMeans(n_clusters=6).fit(test_data)
        ri= randIndex(test_labels, km.labels_)
        print(ri)
        if ri > maximum:
            maximum = ri
    print(ri)
    '''
    
    '''
    maximum = 0
    p = 0
    

    for i in range(1000):
        km = KMeans(n_clusters=6).fit(test_data)
        p= purity(km.labels_,test_labels)
        #print(p)
        if p > maximum:
            maximum = p
    print('Purity Maximum: ',maximum)
    '''
    
    '''
    maximum = 0
    asc = 0
    

    for i in range(1000):
        km = KMeans(n_clusters=6).fit(test_data)
        asc= silhouette_score(test_data, labels = km.labels_)
        #print(p)
        if asc > maximum:
            maximum = asc
    print('ASC Maximum: ',maximum)
    '''
    
    
    
    agg = AgglomerativeClustering(n_clusters=6).fit(test_data)
    clustering = DBSCAN(eps= 0.3, min_samples= 10).fit(test_data)
    
    #print(clustering.labels_)
    
    #---Rand Index
    print("Rand Index ")
    
    #Kmeans
    print('Kmeans Rand Index: ',randIndex(test_labels, km.labels_))
    
    #Agg
    print('Agglomerative Rand Index: ',randIndex(test_labels, agg.labels_))
    
    #DBSCAN

    print('DBSCAN Rand Index: ',randIndex(test_labels, clustering.labels_))
    
    print("-----------------------------")
    
    #---Purity
    print("Purity")
    
    #Kmeans
    print('Kmeans Purity: ',purity(km.labels_, test_labels))
    
    #Agg
    print('Agglomerative Purity: ',purity(agg.labels_, test_labels))
    
    #DBSCAN
    print('DBSCAN Purity: ',purity(clustering.labels_, test_labels))
    print("-----------------------------")
    
    #---Silhouette
    print("Silhouette")
    
    #Kmeans
    print('Kmeans Silhouette: ',silhouette_score(test_data, labels = km.labels_) )
    
    #Agg
    print('Agglomerative Silhouette: ',silhouette_score(test_data, labels = agg.labels_))
    
    #DBSCAN (last two digits different)
    print('DBSCAN Silhouette: ', silhouette_score(test_data, labels = clustering.labels_))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    