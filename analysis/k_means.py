"""

"""

from random import randint
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt

class Centroid:
    #TODO Add description
    def __init__(self, coordinate_size, volume):
        #Sets the initial centroid position at a random coordinate
        self.coords = np.zeros(coordinate_size)
        for i in range(len(self.coords)):
            self.coords[i] = randint(0, volume)
    
    def set_coords(self, coordinates):
        self.coords = coordinates

class Vector:
    #TODO Add description
    def __init__(self, coordinates):
        self.coords = coordinates
        self.belongs_to = None
    
    def set_belongs_to(self, centroid):
        self.belongs_to = centroid
        
def k_means(vectors, volume, num_centroids):
    """
    
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
    
    return centroids
        

def find_distance(a, b):
    #TODO Add description
    if len(a) != len(b):
        return "ERROR"
    else:
        total = 0
        for i in range(len(a)):
            total += (a[i] - b[i])**2
        
        return sqrt(total)
            
if __name__ == "__main__":
    vectors = np.array([np.array([randint(0, 8*8*7), randint(0, 8*8*7), randint(0, 8*8*7)]) for _ in range(100)])
    centroids = k_means(vectors, 8*8*7, 3)

    fig = plt.figure()
    x = [item[0] for item in vectors]
    y = [item[1] for item in vectors]
    z = [item[2] for item in vectors]

    x_centroid = [item.coords[0] for item in centroids]
    y_centroid = [item.coords[1] for item in centroids]
    z_centroid = [item.coords[2] for item in centroids]

    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z)
    ax.scatter(x_centroid, y_centroid, z_centroid)
    plt.show()