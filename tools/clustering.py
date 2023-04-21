"""
Module to implement various clustering algorithms for clustering of data 
using unsupervied machine learning
"""
#%%
from typing import Tuple
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def choose_num_clusters(
    data: np.array, 
    num_clusters = 20,
    plot: bool = False
    ) -> Tuple:
    """ 
    Chooses the optimal number of clusters for k_means clustering 
    by choosing the number of clusters which 
    maximisises the silhouette coefficient
    
    :param data: data to find the optimal number of clusters for
    :param plot: boolean indicating if the silhouette coefficient trend should be plotted
    :rtype: Tuple
    :return: the optimal number of clusters and the labels produced by the clustering
    """
    silhouette_scores = np.zeros(num_clusters - 2) # List of silhouette scores
    max_silouette = 0 
    optimal_num_clusters = 0
    optimal_clustering_labels = None
    
    # Iterates through the potential number of clusters
    for num in range(2, num_clusters):
        k_means = KMeans(n_clusters=num).fit(data) # performs k-means with the data using specified number of clusters
        silhouette_coeff = silhouette_score(data, k_means.labels_) # Calculates the mean silhouette coefficient of the clustering
        silhouette_scores[num-2] = silhouette_coeff # Appends silhouette coefficient to list
        
        # Checks if silhouette coefficient is greater than the current maximum
        if silhouette_coeff > max_silouette:
            optimal_num_clusters = num
            max_silouette = silhouette_coeff
            optimal_clustering_labels = k_means.labels_
    
    if plot: # If plotting is enabled plot the silhouette coefficient at each clustering number
        sns.set_style("darkgrid")
        plt.xlabel("Number of clusters")
        plt.ylabel("Mean Silhouette Coefficient")
        plt.plot(silhouette_scores)
        plt.xticks(np.arange(len(silhouette_scores)), np.arange(2, num_clusters)) # Starts the x axis from 2 clusters
        plt.show()
    
    return optimal_num_clusters, optimal_clustering_labels, max_silouette


#%%