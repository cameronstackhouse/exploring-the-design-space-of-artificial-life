import time
import logging #TODO Use this
import subprocess as sub
from voxcraftpython.VoxcraftVXA import VXA
from voxcraftpython.VoxcraftVXD import VXD

#TODO Finish function to evaluate a population of organisms
#TODO Add input option to specify what is being tested for (Locomotion, Object Movement, Object Transport)

def evaluate_pop(pop, run_directory, run_name, truncation_rate):
    """
    
    """
    start = time.time() 
    num_evaluated = 0

    vxa = VXA(EnableExpansion=1, SimTime=5) # pass vxa tags in here
    vxa.write(f"{run_directory}/fitnessFiles/base.vxa") #Writes the base for a generation of simulations

    #Iterates through the population to evaluate each one individually
    for n, individual in enumerate(pop):
        body = individual.to_phenotype() #Gets the phenotype generated by the CPPN
        vxd = VXD()
        vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
        vxd.set_data(body) #Sets the data to be written as the phenotype generated
        vxd.write(f"{run_directory}/fitnessFiles/{run_name}--id_{n}.vxd") #Writes vxd file of current individual to the run directory

        #TODO Evaluate using voxcraft-sim, checking for errors
        sub.Popen(f"./voxcraft-sim -f {run_directory}/fitnessFiles/{run_name}--id{n}.vxd", shell=True)

        #TODO Read results from history file and set the CPPNs fitness to that value
    
    time_taken = time.time() - start #Time taken to evaluate one generation