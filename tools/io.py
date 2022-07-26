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
    #TODO
    results = []
    tree = ET.parse(f"{filename}.xml")
    root = tree.getroot()
    for child in root:
        individual_result = dict()
        #TODO Can access different tags by child[index]
        #TODO Add to individual result dictionary
        results.append(individual_result)

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

read_sim_output("tools/test")