import time
import subprocess as sub
from voxcraftpython.VoxcraftVXA import VXA
from voxcraftpython.VoxcraftVXD import VXD

#TODO

def evaluate_pop(pop, run_directory, run_name):
    """
    
    """
    start = time.time()
    num_evaluated = 0

    vxa = VXA(EnableExpansion=1, SimTime=5) # pass vxa tags in here
    vxa.write("base.vxa") #Writes


    for n, individual in enumerate(pop):
        body = individual.to_phenotype()
        vxd = VXD()
        vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
        vxd.set_data(body)
        vxd.write(f"{run_name}--id_{n}.vxd")    # Write out the vxd to data/

        #TODO Evaluate using voxcraft-sim, checking for errors
        num_evaluated += 1

        #TODO Add voxcraft-sim executable to file
        sub.Popen(f"./voxcraft-sim -f {run_directory}/simulationFiles/{run_name}--id{n}.vxa", shell=True)

        #TODO Add logging