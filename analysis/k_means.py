"""

"""

from random import randint
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt

class Centroid:
    """
    Class to represent a centroid in the k-means
    algorithm
    """
    #TODO Add description
    def __init__(self, coordinate_size, volume):
        """
        
        """
        #Sets the initial centroid position at a random coordinate
        self.coords = np.zeros(coordinate_size)
        for i in range(len(self.coords)):
            self.coords[i] = randint(0, volume)
    
    def set_coords(self, coordinates):
        """
        
        """
        self.coords = coordinates

class Vector:
    """
    
    """
    #TODO Add description
    def __init__(self, coordinates):
        self.coords = coordinates
        self.belongs_to = None
    
    def set_belongs_to(self, centroid):
        """
        
        """
        self.belongs_to = centroid
        
def k_means(vectors, volume, num_centroids):
    """
    Function to find centroid positions given a list of vectors,
    a coordinate space, and a given number of centroids using 
    the k-means algorithm

    :param vectors:
    :param volume:
    :param num_centroids:
    :rtype:
    :return:
    """
    #TODO Add description and comments
    vector_size = len(vectors[0])
    vector_objects = []
    for vector in vectors:
        vector_objects.append(Vector(vector))

    centroids = [Centroid(vector_size, volume) for _ in range(num_centroids)] #Sets the number of centroids to the number of different types of organisms

    dif = None
    while dif is None or dif != 0:
        dif = 0
        for vector in vector_objects:
            results = []
            for i in range(num_centroids):
                results.append(find_distance(vector.coords, centroids[i].coords))
            
            belongs_to = results.index(min(results))

            if vector.belongs_to is not centroids[belongs_to]:
                dif += 1

            vector.set_belongs_to(centroids[belongs_to]) 

            for centroid in centroids:
                counter = 0
                total = np.array([0 for _ in range(vector_size)])
                for vector in vector_objects:
                    if vector.belongs_to is centroid:
                        counter+=1
                        total = total + vector.coords
                
                if counter == 0:
                    continue
                else:
                    centroid.set_coords(np.divide(total, counter))
        
        print(dif)
    
    return centroids, vector_objects
        

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

def plot(centroids, vectors, three_dimensional: bool = False) -> None:
    """
    Function to plot centroids and vectors using matplotlib

    :param centroids:
    :param vectors:
    :param three_dimensional:
    """
    #TODO Add comments and description
    fig = plt.figure()
    if three_dimensional:
        ax = fig.add_subplot(projection="3d")
    else:
        ax = fig.add_subplot()
    for i in range(len(centroids)):
        if three_dimensional:
            x = [item.coords[0] for item in vectors if item.belongs_to == centroids[i]]
            y = [item.coords[1] for item in vectors if item.belongs_to == centroids[i]]
            z = [item.coords[2] for item in vectors if item.belongs_to == centroids[i]]
            ax.scatter(x, y, z)
        else:
            x = [item.coords[0] for item in vectors if item.belongs_to == centroids[i]]
            y = [item.coords[1] for item in vectors if item.belongs_to == centroids[i]]
            ax.scatter(x, y)
    
    if three_dimensional:
        x_centroid = [item.coords[0] for item in centroids]
        y_centroid = [item.coords[1] for item in centroids]
        z_centroid = [item.coords[2] for item in centroids]
        ax.scatter(x_centroid, y_centroid, z_centroid, color=['black'])
    else:
        x_centroid = [item.coords[0] for item in centroids]
        y_centroid = [item.coords[1] for item in centroids]
        ax.scatter(x_centroid, y_centroid, color=['black'])
    
    plt.show()

if __name__ == "__main__":
    vectors = np.array([np.array([randint(0, 8*8*7), randint(0, 8*8*7)]) for _ in range(500)])
    centroids, vectors = k_means(vectors, 8*8*7, 3)
    plot(centroids, vectors)