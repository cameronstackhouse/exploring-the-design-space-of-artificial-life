"""
Module allowing for a population of xenobots to have their fitness analysed
by voxcraft-sim (https://github.com/voxcraft/voxcraft-sim).
"""

import os 
import neat
import numpy as np
import tools.fitness_functions
from typing import List
from tools.read_files import read_sim_output
from tools.gen_phenotype_from_genotype import genotype_to_phenotype
from pureples.es_hyperneat.es_hyperneat import ESNetwork

#Imports an interface for writing VXA and VXD files
from voxcraftpython.VoxcraftVXA import VXA
from voxcraftpython.VoxcraftVXD import VXD

#TODO Add input option to specify what is being tested for (Locomotion, Object Movement, Object Transport)

class Run:
    """
    Container 
    """
    def __init__(
        self, 
        name: str, 
        fitness_func = None,
        params = None, 
        substrate = None, 
        size_params: List = [8,8,7], 
        hyperneat: bool = False
        ) -> None:
        """
        Initilises a run object for running evaluations of 
        xenobots
        """
        self.generation = 0
        self.name = name
        self.params = params
        self.substrate = substrate
        self.size_params = size_params
        self.hyperneat = hyperneat
        self.fitness_function = fitness_func
    
    def evaluate(
        self, 
        genomes, 
        config
        ) -> None:
        """
        Function to evaluate a population of xenobots using 
        voxcraft-sim and assign each xenobot a fitness
        
        :param genomes:
        :param config: configuration file for run
        """
        
        # TODO: THIS IS WHERE FITNESS FUNCTION IS CHANGED!
        # vxa.set_fitness_function(self.fitness_function)
        vxa = VXA(SimTime=3, HeapSize=0.65, RecordStepSize=100, DtFrac=0.95, EnableExpansion=1) 
    
        #Adds both cardiac, skin cells, and non-xenobot cells to the simulation
        passive = vxa.add_material(RGBA=(0,255,0), E=5000000, RHO=1000000) # passive soft
        active = vxa.add_material(RGBA=(255,0,0), CTE=0.01, E=5000000, RHO=1000000) # active
        fixed = vxa.add_material(RGBA=(0,0,0)) # Used for objects in environment #TODO: SET FIXED == TRUE
        moveable = vxa.add_material(RGBA=(128,128,128), E=5000000, RHO=1000000) # Used for objects in environment which xenobots can move
    
        os.system(f"rm -rf fitnessFiles/{self.name}/{self.generation}") #Deletes contents of run directory if exists

        os.system(f"mkdir -p fitnessFiles/{self.name}/{self.generation}") # Creates a new directory to store fitness files
    
        vxa.write("base.vxa") #Write a base vxa file
    
        os.system(f"cp base.vxa fitnessFiles/{self.name}/{self.generation}") #Copy vxa file to correct run directory
        os.system("rm base.vxa") #Removes old vxa file

        for id, (_, genome) in enumerate(genomes):
            net = None
            if self.hyperneat:
                cppn_designer = neat.nn.FeedForwardNetwork.create(genome, config) # CPPN Which designs network to create xenobot in HyperNEAT
                xenobot_producer_network = ESNetwork(self.substrate, cppn_designer, self.params) # CPPN designed by HyperNEAT CPPN to produce xenobot
                net = xenobot_producer_network.create_phenotype_network()
            else:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            body_size = 1
            for dim in self.size_params:
                body_size *= dim
            body = np.zeros(body_size)
            
            body = genotype_to_phenotype(net, self.size_params) # Generates xenobot body using CPPN

            for cell in range(len(body)):
                if body[cell] == 1:
                    body[cell] = passive
                elif body[cell] == 2:
                    body[cell] = active 
        
            body = body.reshape(self.size_params[0],self.size_params[1],self.size_params[2])
            # TODO: Add obsticles in VXD file if correct fit func
            # TODO: Record voxel for object expulsion
            vxd = VXD()
            
            if self.fitness_function == tools.fitness_functions.object_expulsion:
                object = np.zeros([8,8,7])
                # TODO WORK OUT HOW TO TRACK MOVEMENT OF THIS OBJECT
                object[4,4,1] = moveable
                np.append(body, object)
            elif self.fitness_function == tools.fitness_functions.tall_obsticle:
                # Creates wall and adds it to xenobot structure
                wall = np.zeros([8,8,20])
                wall_row = 20
                for axis_1_cell in wall[0]:
                    for axis_2_cell in wall[1]:
                      wall[axis_1_cell][axis_2_cell][wall_row] = fixed
                np.append(body, wall)
            
            vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
            vxd.set_data(body) #Sets the data to be written as the phenotype generated

            vxd.write(f"id{id}.vxd") #Writes vxd file for individual
            os.system(f"cp id{id}.vxd fitnessFiles/{self.name}/{self.generation}")
            os.system(f"rm id{id}.vxd") #Removes the old non-copied vxd file
    
        os.chdir("voxcraft-sim/build") # Changes directory to the voxcraft directory TODO change to be taken from settings file
        os.system(f"./voxcraft-sim -i ../../fitnessFiles/{self.name}/{self.generation} -o ../../fitnessFiles/{self.name}/{self.generation}/output.xml -f > ../../fitnessFiles/{self.name}/{self.generation}/test.history")
        os.chdir("../../") # Return to project directory
    
        results = read_sim_output(f"fitnessFiles/{self.name}/{self.generation}/output") #Reads sim results from output file

        #Remove history file
        os.system(f"rm -f fitnessFiles/{self.name}/{self.generation}/test.history")
        
        # Assigns fitness to individuals based on results
        for result in results:
            genomes[result["index"]][1].fitness = float(result["fitness"])
        
        self.generation += 1