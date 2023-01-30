"""
Module allowing for a population of xenobots to have their fitness analysed
by voxcraft-sim (https://github.com/voxcraft/voxcraft-sim).
"""

import time
import logging
import os 
from typing import List
from tools.read_files import read_sim_output
from tools.fitness_functions import FitnessFunction

#Imports an interface for writing VXA and VXD files
from voxcraftpython.VoxcraftVXA import VXA
from voxcraftpython.VoxcraftVXD import VXD

#TODO Finish function to evaluate a population of organisms
#TODO Add input option to specify what is being tested for (Locomotion, Object Movement, Object Transport)

def evaluate_pop(
    pop: List, 
    run_directory: str, 
    run_name: str, 
    fitness_function: FitnessFunction
    )-> None:
    """
    Function to evaluate a population of computer-designed organisms generated
    by a population of CPPNs using voxcraft-sim

    :param pop: population of CPPNs
    :param run_directory: directory containing all generations of runs
    :param run_name: name of the evaluation run
    :param fitness_function: fitness function to be used for evaluation
    """

    #TODO look at MathTree to see how to create fitness function!
    #TODO Check if fitness func is apropriate
    #TODO Check if limitations should be placed on number of active cells
    vxa = VXA(SimTime=2, HeapSize=0.65, RecordStepSize=100, DtFrac=0.95, EnableExpansion=1) 
    
    #Adds both cardiac and skin cells to the simulation
    passive = vxa.add_material(RGBA=(0,255,0), E=5000000, RHO=1000000) # passive soft
    active = vxa.add_material(RGBA=(255,0,0), CTE=0.01, E=5000000, RHO=1000000) # active

    os.system(f"rm -rf fitnessFiles/{run_directory}/{run_name}/") #Deletes contents of run directory if exists

    os.system(f"mkdir -p fitnessFiles/{run_directory}/{run_name}/") # Creates a new directory to store fitness files

    #Initilises logging
    logging.basicConfig(filename=f"fitnessFiles/{run_directory}/evaluation.log", format='%(levelname)s:%(message)s', level=logging.DEBUG)
    vxa.write("base.vxa") #Write a base vxa file
    os.system(f"cp base.vxa fitnessFiles/{run_directory}/{run_name}/") #Copy vxa file to correct run directory
    os.system("rm base.vxa") #Removes old vxa file

    #Iterates through the population and writes a vxd file for each
    logging.info(f"Writing {len(pop)} vxd files for xenobots")
    for n, individual in enumerate(pop):
        body = individual.to_phenotype() #Gets the phenotype generated by the CPPN

        for cell in range(len(body)):
            if body[cell] == 1:
                body[cell] = passive
            elif body[cell] == 2:
                body[cell] = active 
        
        body = body.reshape(8,8,7)
        vxd = VXD()
        vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
        vxd.set_data(body) #Sets the data to be written as the phenotype generated

        vxd.write(f"id{n}.vxd") #Writes vxd file for individual
        os.system(f"cp id{n}.vxd fitnessFiles/{run_directory}/{run_name}/")
        os.system(f"rm id{n}.vxd") #Removes the old non-copied vxd file

    #Uses voxcraft-sim to evaluate populations fitness, producing an output xml file and a history file, which can be visualised by voxcraft-viz
    logging.info("Simulating...")

    os.chdir("voxcraft-sim/build") # Changes directory to the voxcraft directory TODO change to be taken from settings file
    os.system(f"./voxcraft-sim -i ../../fitnessFiles/{run_directory}/{run_name}/ -o ../../fitnessFiles/{run_directory}/{run_name}/output.xml -f > ../../fitnessFiles/{run_directory}/{run_name}/{run_name}.history")
    os.chdir("../../") # Return to project directory

    logging.info("Finished simulation")
    results = read_sim_output(f"fitnessFiles/{run_directory}/{run_name}/output") #Reads sim results from output file
    
    #TODO Change this to assign fitnesses correctly :)
    #Sets the fitness of each phenotype using results obtained    
    for result in results:
        population[result["index"]] = float(result["fitness"])

    logging.info(f"Evaluation complete for generation {run_name}.")
