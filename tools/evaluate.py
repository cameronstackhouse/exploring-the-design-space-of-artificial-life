import time
import subprocess as sub
from voxcraftpython.VoxcraftVXA import VXA
from voxcraftpython.VoxcraftVXD import VXD

#TODO

def evaluate_pop(pop, run_directory, run_name, truncation_rate):
    """
    
    """
    start = time.time()
    num_evaluated = 0

    vxa = VXA(EnableExpansion=1, SimTime=5) # pass vxa tags in here
    vxa.write(f"{run_directory}/fitnessFiles/base.vxa") #Writes the base for a generation of simulations


    for n, individual in enumerate(pop):
        body = individual.to_phenotype()
        vxd = VXD()
        vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
        vxd.set_data(body)
        vxd.write(f"{run_directory}/fitnessFiles/{run_name}--id_{n}.vxd") #Writes vxd file of current individual to the run directory

        #TODO Evaluate using voxcraft-sim, checking for errors
        num_evaluated += 1

        #TODO Add voxcraft-sim executable to file
        sub.Popen(f"./voxcraft-sim -f {run_directory}/fitnessFiles/{run_name}--id{n}.vxd", shell=True)

        #TODO Add logging