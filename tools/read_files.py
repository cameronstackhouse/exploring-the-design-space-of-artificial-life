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

def read_history(filename: str) -> dict:
    """
    """
    #TODO Add read_history function to read history files to get
    # movement information for fourier transform

    # C++ code to Gen history file
    # if (d_v3->RecordStepSize) { // output History file
    #             if (j % real_stepsize == 0) {
    #                 if (d_v3->RecordVoxel) {
    #                     // Voxels
    #                     printf("<<<Step%d Time:%f>>>", j, d_v3->currentTime);
    #                     for (int i = 0; i < d_v3->num_d_surface_voxels; i++) {
    #                         auto v = d_v3->d_surface_voxels[i];
    #                         if (v->removed)
    #                             continue;
    #                         if (v->isSurface()) {
    #                             printf("%.1f,%.1f,%.1f,", v->pos.x * vs, v->pos.y * vs, v->pos.z * vs);
    #                             printf("%.1f,%.2f,%.2f,%.2f,", v->orient.AngleDegrees(), v->orient.x, v->orient.y, v->orient.z);
    #                             VX3_Vec3D<double> ppp, nnn;
    #                             nnn = v->cornerOffset(NNN);
    #                             ppp = v->cornerOffset(PPP);
    #                             printf("%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,", nnn.x * vs, nnn.y * vs, nnn.z * vs, ppp.x * vs, ppp.y * vs,
    #                                    ppp.z * vs);
    #                             printf("%d,", v->mat->matid); // for coloring
    #                             printf("%.1f,", v->localSignal);  // for coloring as well.
    #                             printf(";");
    #                         }
    #                     }
    #                     printf("<<<>>>");
    #                 }
                
    file = open(filename, "r")

    lst = file.readlines()

    steps = lst[14:-3] # Gets step information

    # Removes 
    for m, step in enumerate(steps):
        end_chev_index = 0
        for n in range(len(step)):
            if step[n] == '>':
                end_chev_index = n+3
                break
        
        steps[m] = step[end_chev_index:-7]
    
    surface_voxels = steps[1].split(";")

    print(surface_voxels[200].split(','))

    #Each split line: surface voxel info
    #Format: pos.x,y,z * vs, orientation, xori, yori, zori, nnn.x,y,z , ppp.x,y,z, colouring, colouring



def read_vxd(filename: str):
    """
    """
    pass

if __name__ == "__main__":
    #TODO DELETE FOR RELEASE
    read_history("demo_basic2.history")