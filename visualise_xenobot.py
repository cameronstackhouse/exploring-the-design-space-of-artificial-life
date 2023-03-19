""" 
Module containing code to visualise xenobot structures in python
"""
# %%
import matplotlib.pyplot as plt
from typing import List
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 

def show(
    body: tuple, 
    dimensions: np.array = [8,8,7]
    ) -> None:
    """
    :param body: string representing xenobot body
    :param dimensions: list describing xenobot 
    """
    str_body = ""
    for cell in list(body):
        str_body += str(int(cell))
    
    array_body = np.zeros(len(str_body))
    
    for i in range(len(array_body)):
        array_body[i] = str_body[i]
    
    body = array_body.reshape(dimensions[0],dimensions[1],dimensions[2])
    
    alpha = 0.9
    colours = np.empty(dimensions + [4], dtype=np.float32)
    colours[body==1] = [1, 0, 0, alpha]
    colours[body==2] = [0, 0, 1, alpha]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    ax.voxels(body, facecolors=colours, edgecolors="black")
    
    plt.show()

if __name__ == "__main__":
    body = np.zeros(8*8*7)
    
    for i in range(8*8*7):
        body[i] = 1
    
    show(body)
    
    