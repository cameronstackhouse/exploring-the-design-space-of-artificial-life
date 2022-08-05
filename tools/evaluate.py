import time
import logging #TODO Use this
import subprocess as sub
from typing import List
from enum import Enum
from read_outputs import read_sim_output

#Imports an interface for writing VXA and VXD files
from voxcraftpython.VoxcraftVXA import VXA
from voxcraftpython.VoxcraftVXD import VXD

#TODO Finish function to evaluate a population of organisms
#TODO Add input option to specify what is being tested for (Locomotion, Object Movement, Object Transport)

class FitnessFunction(Enum):
    MAX_DISTANCE = 1
    VERTICAL_DISTANCE = 2
    INTERACTION = 3
    OBJECT_EXPULSION = 4
    #TODO Add more fitness func values here

def evaluate_pop(pop: List, run_directory: str, run_name: str, fitness_function: FitnessFunction):
    """
    Function to evaluate a population of computer-designed organisms generated
    by a population of CPPNs using voxcraft-sim

    :param pop: population of CPPNs
    :param run_directory: directory containing all generations of runs
    :param run_name: name of the evaluation run
    :param fitness_function: fitness function to be used for evaluation
    """
    #Initilises logging
    logging.basicConfig(filename=f"{run_name}_evaluation.log", format='%(levelname)s:%(message)s', encoding="utf-8", level=logging.DEBUG)

    start = time.time() #Starts a timer

    #TODO look at MathTree to see how to create fitness function!
    vxa = VXA() #TODO pass vxa tags in here
    
    #Adds both cardiac and skin cells to the simulation
    vxa.add_material(RGBA=(255,0,255), E=5e4, RHO=1e4)
    vxa.add_material(RGBA=(255,0,0), E=1e8, RHO=1e4)

    sub.Popen(f"rm -r /fitnessFiles/{run_directory}/{run_name}", shell=True) #Deletes contents of run directory if exists

    vxa.write("base.vxa")
    sub.Popen(f"cp base.vxa /fitnessFiles/{run_directory}/{run_name}/", shell=True) #TODO Ensure works!

    #Iterates through the population to evaluate each one individually
    for n, individual in enumerate(pop):
        body = individual.to_phenotype() #Gets the phenotype generated by the CPPN
        vxd = VXD()
        vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
        vxd.set_data(body) #Sets the data to be written as the phenotype generated

        vxd.write(f"id{n}.vxd") 
        sub.Popen(f"cp id{n}.vxd /fitnessFiles/{run_directory}/{run_name}/", shell=True) #TODO Ensure works!

        logging.info(f"Writing vxd file for individual: {n}")

    #TODO Check for errors maybe?
    #TODO Ensure this works!

    #Uses voxcraft-sim to evaluate populations fitness
    sub.Popen(f"./voxcraft-sim -i /fitnessFiles/{run_directory}/{run_name}/ -o /fitnessFiles/{run_directory}/{run_name}/output.xml > /fitnessFiles/{run_directory}/{run_name}/{run_name}.history", shell=True)
    
    results = read_sim_output(f"/fitnessFiles/{run_directory}/{run_name}/output") #Reads sim results from output file
    
    #Sets the fitness of each phenotype using results obtained
    for n, indv in enumerate(results):
        pop[n].fitness = float(indv["fitness"]) 

    time_taken = time.time() - start #Time taken to evaluate one generation
    logging.info(f"Evaluation complete. Time taken: {time_taken}.") 
