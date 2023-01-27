"""
Module containing I/O functions for reading/writing
to various different files required for running simulations and experiments
"""

import json
from typing import List
import xml.etree.ElementTree as ET

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


if __name__ == "__main__":
    #TODO DELETE FOR RELEASE
    q = read_sim_output("tools/example")
    b = read_settings("settings")
    print(b)