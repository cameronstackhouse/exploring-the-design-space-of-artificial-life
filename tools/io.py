"""
Module containing I/O functions for reading/writing
to various different files
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
    #TODO Add comments and expand upon this
    results = []
    tree = ET.parse(f"{filename}.xml")
    root = tree.getroot()
    
    bestfit = root[1]
    detail = root[2]

    for robot in detail:
        result = dict()
        fitness = robot[1].text
        current_x = robot[7][0].text
        current_y = robot[7][1].text
        current_z = robot[7][2].text
        total_distance_of_all_voxels = robot[8].text
        
        result["fitness"] = fitness
        result["current_x"] = current_x
        result["current_y"] = current_y
        result["current_z"] = current_z
        result["total_distance"] = total_distance_of_all_voxels

        results.append(result)

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

q = read_sim_output("tools/example")

print(q)