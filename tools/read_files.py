  #%%

"""
Module containing I/O functions for reading/writing
to various different files required for running simulations and experiments
"""

import json
from typing import List
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftn

def read_sim_output(filename: str) -> List[dict]:
    """
    Function to read the output of voxcraft-sim and store the
    values for each individual in a dictionary

    :param filename: name of the xml file to read
    :rtype: List of dictionaries
    :return: List of dictionaries each containing simulation information for each individual
    """
    #TODO expand upon this
    results = []
    tree = ET.parse(f"{filename}.xml") #Parses the XML file
    root = tree.getroot() #Gets the root of the document
    
    bestfit = root[1] #Accesses the best fit section of the xml file
    detail = root[2] #Accesses the details section of the xml file

    #Iterates through all robots details in the simulation
    for robot in detail:
        result = dict()
        fitness = robot[1].text #Gets the fitness of the voxel
        current_x = robot[7][0].text #Gets the new x position of the voxel
        current_y = robot[7][1].text #Gets the new y position of the voxel
        current_z = robot[7][2].text #Gets the new z position of the voxel
        total_distance_of_all_voxels = robot[8].text 
        
        result["index"] = int((robot.tag)[2:]) # Gets the number of the xenobot evaluated
        result["fitness"] = fitness
        result["x_movement"] = float(current_x) - float(robot[6][0].text) #Finds the distance the voxel has moved along the x axis
        result["y_movement"] = float(current_y) - float(robot[6][1].text) #Finds the distance the voxel has moved along the y axis
        result["z_movement"] = float(current_z) - float(robot[6][2].text) #Finds the distance the voxel has moved along the z axis
        result["total_distance"] = float(total_distance_of_all_voxels)

        results.append(result) #Adds the results of the individual to a list of results

    return results

def read_settings(filename: str) -> dict:
    """
    Function to read a JSON file into a 
    dictionary

    :param filename: name of the JSON file to read
    :rtype: dict
    :return: dictionary representation of the JSON file
    """
    f = open(f"{filename}.json") #Opens the json file
    data = json.load(f) #Loads the json file into a dictionary
    f.close()

    return data

def read_history(
    filename: str
    ) -> dict:
    """
    Function to read the history file produced by voxcraft-sim for the movement
    of a xenobot, allowing for the fourier transform of the movement of the 
    xenobot to be calculated.

    :param filename: 
    """
    #TODO Add read_history function to read history files to get
    # movement information for fourier transform

    #TODO Tidy and add comments
    x_components = []
    y_components = []
    z_components = []

    file = open(filename, "r")

    lst = file.readlines()

    # Search for steps start index for start of steps
    start_index = 0
    for i in range(len(lst)):
        if lst[i][:3] == "<<<":
            start_index = i
            break
    
    steps = lst[start_index:-3] # Gets step information

    # Removes uneeded information from step data
    for m, step in enumerate(steps):
        end_chev_index = 0
        for n in range(len(step)):
            if step[n] == '>':
                end_chev_index = n+3
                break
        
        steps[m] = step[end_chev_index:-7]
    
    #Each split line: surface voxel info
    #Format: pos.x,y,z * vs, orientation, xori, yori, zori, nnn.x,y,z , ppp.x,y,z, colouring, colouring
    
    # Iterates through each step getting the position of a measured voxel at each timestep
    for step in steps:
        surface_voxels = step.split(";")
        total_x = 0
        total_y = 0
        total_z = 0
        for voxel in surface_voxels:
            v = voxel.split(",")
            if v != ['']:
                total_x += float(v[0])
                total_y += float(v[1])
                total_z += float(v[2])
            
        x_components.append(total_x/len(surface_voxels))
        y_components.append(total_y/len(surface_voxels))
        z_components.append(total_z/len(surface_voxels))

    x_components = np.array(x_components)
    y_components = np.array(y_components)
    z_components = np.array(z_components)
    
    x_components[1:] -= x_components[:-1].copy()
    y_components[1:] -= y_components[:-1].copy()
    z_components[1:] -= z_components[:-1].copy()
    return [x_components[1:], y_components[1:], z_components[1:]]

def read_xenobot_from_vxd(filename: str) -> np.array:
    """ 
    Function to read a xenobot body from a vxd file
    
    :param filename: name of xenobot file to open
    :rtype: np.array
    :return: 
    """
    layer_data = []
    x_voxels = 0
    y_voxels = 0
    z_voxels = 0
    
    # Splits file into lines
    with open(filename) as f:
        lines = f.read().splitlines()
    
    # Strips lines of whitespace
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
    
    # TODO Remove walls and fixed objects
    # Gets xenobot structure data
    for line in lines:
        if line[:7] == "<Layer>":
            layer_data.append(line[16:-11])
        elif line[:10] == "<X_Voxels>":
            x_voxels = int(line[10])
        elif line[:10] == "<Y_Voxels>":
            y_voxels = int(line[10])
        elif line[:10] == "<Z_Voxels>":
            z_voxels = int(line[10])
        
    # Splits into 2D array of integers
    for i in range(len(layer_data)):
        layer_data[i] = [int(s) for s in layer_data[i]]
    
    # Assemble flattened array into 3D xenobot structure 
    body = np.zeros((x_voxels, y_voxels, z_voxels), dtype=np.int8)
    
    for z in range(len(layer_data)):
        k = 0
        for y in range(y_voxels):
            for x in range(x_voxels):
                body[x, y, z] = layer_data[z][k]
                k += 1
    
    # Reconstruct xenobot from layer data
    return body

if __name__ == "__main__":
    test = read_history("demo_basic.history")
    
    plt.plot(test[0])
    
    fft_d = fftn(test)
    
    print(len(fft_d))
    
    
    
   

# %%
