"""
Module to implement the k-means algorithm for clustering of data 
using unsupervied machine learning
"""

from random import randint
from math import sqrt
from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt

class Centroid:
    """
    Class to represent a centroid in the k-means
    algorithm
    """
    def __init__(self, coordinate_size: int, axis_sizes: List[int]) -> None:
        """
        Method to initilise a centroid with a random coordinate position

        :param coordinate_size: number of coordinates in the vector
        :param axis_sizes: list of the size of each axis, for example [8,8,7]: x = 8, y = 8, z = 7
        """
        #Sets the initial centroid position at a random coordinate
        self.coords = np.zeros(coordinate_size) #Empty array of 0 the same size as the vectors to be plotted against
        #Populates new empty vector with random values
        for i in range(len(self.coords)):
            self.coords[i] = randint(0, axis_sizes[i]) #Chooses a random number between 0 and the max size of the current axis
    
    def set_coords(self, coordinates: List[int]) -> None:
        """
        Method to set the coordinates of a centroid

        :param coordinates: coordinates to place the centroid
        """
        self.coords = coordinates

class Vector:
    """
    Class to represent a vector to plot in the k-means
    algorithm
    """
    def __init__(self, coordinates: np.array) -> None:
        """
        Method to initilise a vector with a set of coordinates
        and no initial centroid that it belongs to

        :param coordinates: coordinates to place the vector
        """
        self.coords = coordinates
        self.belongs_to = None
    
    def set_belongs_to(self, centroid: Centroid) -> None:
        """
        Sets the belongs_to attribute, changing the Centroid that
        a given vector belongs to

        :param centroid: Centroid that the Vector now belongs to
        """
        self.belongs_to = centroid
        
def k_means(vectors: np.array, axis_sizes: List[int], num_centroids: int) -> Tuple[List[Centroid], List[Vector]]:
    """
    Function to find centroid positions given a list of vectors,
    a coordinate space, and a given number of centroids using 
    the k-means algorithm

    :param vectors: numpy array of datapoints to plot
    :param axis_sizes: list of the size of each axis, for example [8,8,7]: x = 8, y = 8, z = 7
    :param num_centroids: number of centroids to use (number of clusters to find in the data)
    :rtype: tuple
    :return: tuple containing a list of Centroids and a list of Vectors
    """
    vector_size = len(vectors[0]) #Gets the dimensions of a Vector in the list of Vectors

    #Checks to ensure all Vectors are of the same dimension size
    for vector in vectors:
        if len(vector) != vector_size:
            return "Error, all vectors must be of the same dimension."
    
    vector_objects = []
    for vector in vectors:
        vector_objects.append(Vector(vector)) #Creates a Vector object from the list of coordinates passed in

    centroids = [Centroid(vector_size, axis_sizes) for _ in range(num_centroids)] #Creates a list of Centroids of size num_centroids

    dif = None #Variable to check if an iteration has resulted in any movement of Vectors between Centroids
    while dif is None or dif != 0: #Repeats until there is no difference after an iteration
        dif = 0
        for vector in vector_objects:
            results = []
            #Iterates through each Vector and finds its Euclidian distance from each Centroid
            for i in range(num_centroids):
                results.append(find_distance(vector.coords, centroids[i].coords))
            
            belongs_to = results.index(min(results)) #Finds the Centroid the Vector belongs to (the one that it is closest to)

            if vector.belongs_to is not centroids[belongs_to]: #Checks if the Vector has changed which Centroid it belongs to
                dif += 1 #If so then the difference counter is iterated

            vector.set_belongs_to(centroids[belongs_to])  #Sets the Vector to belong to the Centroid that is is now closest to

            #Changes the position of Centroids based on the mean coordinate position of each Vector that belongs to it
            for centroid in centroids:
                counter = 0 #Counter to store the number of Vectors that belong to a given Centroid
                total = np.array([0 for _ in range(vector_size)]) #Creates an array to store the total coordinate position of all Vectors that belong to the Centroid

                #Iterates through all Vectors
                for vector in vector_objects:
                    if vector.belongs_to is centroid: #Checks if the Vector is in the current Centroid
                        #If so iterate the counter and add its coordinates to the total
                        counter+=1
                        total = total + vector.coords
                
                if counter == 0:
                    #If no Vectors belong to the Centroid then continue
                    continue
                else:
                    #Otherwise divide the total by the counter and set the Centroids coordinates to the newly calculated position
                    centroid.set_coords(np.divide(total, counter))
        
        print(dif)
    
    return centroids, vector_objects #Returns a list of Centroids and list of Vectors
        

def find_distance(a: np.array, b: np.array) -> int:
    """
    Function to find the euclidian distance between two vectors

    :param a: vector a
    :param b: vector b
    :rtype: int
    :return: euclidian distance between the two vectors
    """

    if len(a) != len(b): #Checks to ensure the vectors are of the same length
        return "ERROR"
    else:
        total = 0
        #Iterates through points in the data, calculating difference at each point
        for i in range(len(a)):
            total += (a[i] - b[i])**2
        
        return sqrt(total)

def plot(centroids: np.array, vectors: np.array, three_dimensional: bool = False) -> None:
    """
    Function to plot centroids and vectors using matplotlib

    :param centroids: list of Centroids to plot
    :param vectors: list of Vectors to plot
    :param three_dimensional: boolean indicating if the plot is 3D
    """

    fig = plt.figure() #Creates a new matplotlib figure
    if three_dimensional:
        #If 3D flag is true then add a 3D subplot
        ax = fig.add_subplot(projection="3d")
    else:
        #Otherwise add a 2D subplot
        ax = fig.add_subplot()
    
    #Plots Vectors
    for i in range(len(centroids)): #Iterates through every Centroid
        if three_dimensional: #If 3D plot x, y, and z
            #Plots a Vector if it belongs to the current Centroid and adds it to the figure
            x = [item.coords[0] for item in vectors if item.belongs_to == centroids[i]]
            y = [item.coords[1] for item in vectors if item.belongs_to == centroids[i]]
            z = [item.coords[2] for item in vectors if item.belongs_to == centroids[i]]
            ax.scatter(x, y, z)
        else: #Else plot x, and y
            #Plots a Vector if it belongs to the current Centroid and adds it to the figure
            x = [item.coords[0] for item in vectors if item.belongs_to == centroids[i]]
            y = [item.coords[1] for item in vectors if item.belongs_to == centroids[i]]
            ax.scatter(x, y)
    
    if three_dimensional: #If 3D plot x, y, and z
        #Plots all Centroids
        x_centroid = [item.coords[0] for item in centroids]
        y_centroid = [item.coords[1] for item in centroids]
        z_centroid = [item.coords[2] for item in centroids]
        ax.scatter(x_centroid, y_centroid, z_centroid, color=['black'])
    else: #Else plot x, and y
        #Plots all Centroids
        x_centroid = [item.coords[0] for item in centroids]
        y_centroid = [item.coords[1] for item in centroids]
        ax.scatter(x_centroid, y_centroid, color=['black'])
    
    plt.show() #Shows the figure

if __name__ == "__main__":
    vectors = np.array([np.array([randint(0, 8), randint(0, 8), randint(0, 8)]) for _ in range(367)])
    centroids, vectors = k_means(vectors, [8,8,7], 7)
    plot(centroids, vectors, True)