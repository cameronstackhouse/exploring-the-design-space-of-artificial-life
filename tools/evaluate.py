import time
import logging #TODO Use this
import subprocess as sub
from typing import List
from tools.read_xml import read_sim_output
from voxcraftpython.VoxcraftVXA import VXA
from voxcraftpython.VoxcraftVXD import VXD

#TODO Finish function to evaluate a population of organisms
#TODO Add input option to specify what is being tested for (Locomotion, Object Movement, Object Transport)

def evaluate_pop(pop, run_directory, run_name, fitness_function) -> List:
    """
    
    """
    #Initilises logging
    logging.basicConfig(filename=f"{run_name}_evaluation.log", format='%(levelname)s:%(message)s', encoding="utf-8", level=logging.DEBUG)

    #TODO Make it return a list of fitness for each phenotype
    start = time.time() 

    fitness = []

    #TODO look at MathTree to see how to create fitness function!
    vxa = VXA() # pass vxa tags in here
    
    #Adds both cardiac and skin cells to the simulation
    vxa.add_material(RGBA=(255,0,255), E=5e4, RHO=1e4)
    vxa.add_material(RGBA=(255,0,0), E=1e8, RHO=1e4)

    sub.Popen(f"rm -r ../fitnessFiles/{run_directory}/{run_name}") #Deletes contents of run directory if exists

    vxa.write("base.vxa") #TODO Copy this to ../fitnessFiles/{run_directory}/{run_name}/

    #Iterates through the population to evaluate each one individually
    for n, individual in enumerate(pop):
        body = individual.to_phenotype() #Gets the phenotype generated by the CPPN
        vxd = VXD()
        vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
        vxd.set_data(body) #Sets the data to be written as the phenotype generated
        vxd.write(f"id{n}.vxd") #TODO Copy to /fitnessFiles/{run_directory}/{run_name}/
        logging.info(f"Writing vxd file for individual: {n}")

    #TODO Evaluate using voxcraft-sim, checking for errors
    #TODO Ensure this works!
    sub.Popen(f"./voxcraft-sim -i ../fitnessFiles/{run_directory}/{run_name}/ -o output.xml > ../fitnessFiles/{run_directory}/{run_name}/{run_name}.history", shell=True)

    #TODO Read results from history file and set the CPPNs fitness to that value
    
    time_taken = time.time() - start #Time taken to evaluate one generation
    logging.info(f"Evaluation complete. Time taken: {time_taken}.")
    
    return fitness