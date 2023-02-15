"""
Module allowing for a population of xenobots to have their fitness analysed
by voxcraft-sim (https://github.com/voxcraft/voxcraft-sim).
"""

import os 
import neat
import numpy as np
import tools.fitness_functions
from tools.read_files import read_sim_output
from tools.activation_functions import normalize
from pureples.es_hyperneat.es_hyperneat import ESNetwork

#Imports an interface for writing VXA and VXD files
from voxcraftpython.VoxcraftVXA import VXA
from voxcraftpython.VoxcraftVXD import VXD

#TODO Add input option to specify what is being tested for (Locomotion, Object Movement, Object Transport)

class Run:
    """
    """
    def __init__(self, name, params = None, substrate = None, size_params=[8,8,7], hyperneat = False):
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
    
    def evaluate(self, genomes, config) -> None:
        """
        Function to evaluate a population of xenobots using 
        voxcraft-sim and assign each xenobot a fitness
        
        :param genomes:
        :param config: configuration file for run
        """
        
        # Gets the inputs for 
        x_inputs = np.zeros(self.size_params)
        y_inputs = np.zeros(self.size_params)
        z_inputs = np.zeros(self.size_params)
    
        for x in range(self.size_params[0]):
                for y in range(self.size_params[1]):
                    for z in range(self.size_params[2]):
                        x_inputs[x, y, z] = x
                        y_inputs[x, y, z] = y
                        z_inputs[x, y, z] = z

        x_inputs = normalize(x_inputs)
        y_inputs = normalize(y_inputs)
        z_inputs = normalize(z_inputs)

        #Creates the d input array, calculating the distance each point is away from the centre
        d_inputs = normalize(np.power(np.power(x_inputs, 2) + np.power(y_inputs, 2) + np.power(z_inputs, 2), 0.5))

        #Creates the b input array, which is just a numpy array of ones
        b_inputs = np.ones(self.size_params)

        #Sets all inputs and flattens them into 1D arrays
        x_inputs = x_inputs.flatten()
        y_inputs = y_inputs.flatten()
        z_inputs = z_inputs.flatten()
        d_inputs = d_inputs.flatten()
        b_inputs = b_inputs.flatten()
    
        inputs = list(zip(x_inputs, y_inputs, z_inputs, d_inputs, b_inputs))

        vxa = VXA(SimTime=3, HeapSize=0.65, RecordStepSize=100, DtFrac=0.95, EnableExpansion=1) 
    
        #Adds both cardiac and skin cells to the simulation
        passive = vxa.add_material(RGBA=(0,255,0), E=5000000, RHO=1000000) # passive soft
        active = vxa.add_material(RGBA=(255,0,0), CTE=0.01, E=5000000, RHO=1000000) # active
    
        os.system(f"rm -rf fitnessFiles/{self.name}/{self.generation}") #Deletes contents of run directory if exists

        os.system(f"mkdir -p fitnessFiles/{self.name}/{self.generation}") # Creates a new directory to store fitness files
    
        vxa.write("base.vxa") #Write a base vxa file
    
        os.system(f"cp base.vxa fitnessFiles/{self.name}/{self.generation}") #Copy vxa file to correct run directory
        os.system("rm base.vxa") #Removes old vxa file

        for id, (_, genome) in enumerate(genomes):
            net = None
            if self.hyperneat:
                #TODO Maybe change this, see if it works
                cppn_designer = neat.nn.FeedForwardNetwork.create(genome, config) # CPPN Which designs network to create xenobot in HyperNEAT
                xenobot_producer_network = ESNetwork(self.substrate, cppn_designer, self.params) # CPPN designed by HyperNEAT CPPN to produce xenobot
                net = xenobot_producer_network.create_phenotype_network()
            else:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            body_size = 1
            for dim in self.size_params:
                body_size *= dim
            body = np.zeros(body_size)
            
            for n, input in enumerate(inputs):
                output = net.activate(input) # Gets output from activating CPPN
                presence = output[0]
                material = output[1]

                if presence <= 0.2: #Checks if presence output is less than 0.2
                    body[n] = 0 #If so there is no material in the location
                elif material < 0.5: #Checks if material output is less than 0.5 
                    body[n] = 1 #If so there is skin in the location
                else:
                    body[n] = 2 #Else there is a cardiac cell in the location

            for cell in range(len(body)):
                if body[cell] == 1:
                    body[cell] = passive
                elif body[cell] == 2:
                    body[cell] = active 
        
            body = body.reshape(self.size_params[0],self.size_params[1],self.size_params[2])
            vxd = VXD()
            vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
            vxd.set_data(body) #Sets the data to be written as the phenotype generated

            vxd.write(f"id{id}.vxd") #Writes vxd file for individual
            os.system(f"cp id{id}.vxd fitnessFiles/{self.name}/{self.generation}")
            os.system(f"rm id{id}.vxd") #Removes the old non-copied vxd file
    
        os.chdir("voxcraft-sim/build") # Changes directory to the voxcraft directory TODO change to be taken from settings file
        os.system(f"./voxcraft-sim -i ../../fitnessFiles/{self.name}/{self.generation} -o ../../fitnessFiles/{self.name}/{self.generation}/output.xml -f > ../../fitnessFiles/{self.name}/{self.generation}/test.history")
        os.chdir("../../") # Return to project directory
    
        results = read_sim_output(f"fitnessFiles/{self.name}/{self.generation}/output") #Reads sim results from output file

        # Assigns fitness to individuals based on results
        for result in results:
            genomes[result["index"]][1].fitness = float(result["fitness"])
        
        self.generation += 1