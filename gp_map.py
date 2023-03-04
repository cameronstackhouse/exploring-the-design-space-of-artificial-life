""" 
Module implementing the ability to generate genotype-phenotype maps
"""
# %%
import os
import pickle
import neat
from copy import deepcopy
from random import choice
from typing import Tuple, List
import numpy as np
import seaborn as sns
from tools.activation_functions import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs
from tools.gen_phenotype_from_genotype import genotype_to_phenotype
from tools.phenotype_information import calc_KC
from tools.read_files import read_sim_output
from visualise_xenobot import show
from voxcraftpython.VoxcraftVXD import VXD
from voxcraftpython.VoxcraftVXA import VXA

# TODO: IMPLEMENT HYPERNEAT GP MAP

class Genotype:
    """ 
    Class to represent a genotype in the genotype map.
    connected_to attribute essential for visualisation of 
    GP map
    """
    def __init__(
        self, 
        geneome,
        id
        ) -> None:
        self.genome = geneome
        self.id = id
        self.phenotype = None

class Phenotype:
    """ 
    Class to represent a phenotype in the genotype-phenotype map
    """
    def __init__(
        self,
        phenotype,
        complexity,
        genotypes
        ) -> None:
        self.phenotype = phenotype
        self.fitness = {} # TODO Implement this
        self.complexity = complexity
        self.genotypes = genotypes

class Connection:
    """ 
    Class for a connection between two genotypes in the 
    genotype space of a GP map
    """
    def __init__(
        self,
        n1: Genotype,
        n2: Genotype
        ) -> None:
        """ 
        Initilises a connection object, specifying the two
        genotypes which are connected
        
        :param n1: Genotype 1
        :param n2: Genotype 2
        """
        self.n1 = n1
        self.n2 = n2

class GenotypePhenotypeMap:
    """ 
    Class representing a genotype-phenotype map of 
    CPPNs and the xenobots created by each
    """
    id_counter = 0
    def __init__(
        self, 
        config_name: str
        ) -> None:
        """ 
        Initilises a genotype-phenotype map object given 
        """
        self.map = {}
        self.phenotypes = []
        self.connections = []
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_name)
        self.config.genome_config.add_activation("neg_abs", neg_abs)
        self.config.genome_config.add_activation("neg_square", neg_square)
        self.config.genome_config.add_activation("sqrt_abs", sqrt_abs)
        self.config.genome_config.add_activation("neg_sqrt_abs", neg_sqrt_abs)
    
    def initilise(self, num_genotypes: int) -> None:
        """ 
        
        """
        self.gen_genotypes(num_genotypes)
        self.gen_phenotype_info()
        self.calculate_fitness()
        
    def gen_genotypes(
        self, 
        num_genotypes: int
        ) -> None:
        """
        Function to generate genotypes... 
        TODO FINISH AND CHANGE TO BE ABLE TO SET SIZE
        
        :param n: size of genotypes to generate (number of nodes)
        """    
        generated_genotypes = set() # Set of generated genotypes
        mutations_applied = set() # Set of genotypes which mutations have been applied to
        
        default_genome = neat.DefaultGenome(1) # Creates a default genome
        default_genome.configure_new(self.config.genome_config) # Configures this default genome
        
        genotype_container = Genotype(default_genome, self.id_counter) # Adds genome to genotype container
        generated_genotypes.add(genotype_container) # Adds the initial genome to the set of generated genotypes
        
        for n in range(num_genotypes):
            # Set of genotypes which havent had mutations applied to them yet
            not_explored = generated_genotypes - mutations_applied
            genotype_container = choice(tuple(not_explored)) # Chooses genotypem from set to apply set of mutations to
        
            # Applies possible range of mutations to the chosen genotype to generate neighbours
            
            # Connection weights
            new_connection_weights = deepcopy(genotype_container) # Creates a copy of the genotype
            # Mutates the connection weights in the 
            for connection in new_connection_weights.genome.connections.values():
                connection.mutate(self.config.genome_config)
        
            generated_genotypes.add(new_connection_weights)
            self.connections.append(Connection(genotype_container, new_connection_weights))
            
            GenotypePhenotypeMap.id_counter += 1
            new_connection_weights.id = GenotypePhenotypeMap.id_counter 
            
            # Node activation functions and bias
            new_activation_functions = deepcopy(genotype_container)
            for node in new_activation_functions.genome.nodes.values():
                node.mutate(self.config.genome_config)
        
            generated_genotypes.add(new_activation_functions)
            self.connections.append(Connection(genotype_container, new_activation_functions))
            GenotypePhenotypeMap.id_counter += 1
            new_activation_functions.id = GenotypePhenotypeMap.id_counter 
            
            # Add connections
            new_connections = deepcopy(genotype_container)
            new_connections.genome.mutate_add_connection(self.config.genome_config)
        
            generated_genotypes.add(new_connections)
            self.connections.append(Connection(genotype_container, new_connections))
            GenotypePhenotypeMap.id_counter += 1
            new_connections.id = GenotypePhenotypeMap.id_counter 
            
            # Add node
            new_node = deepcopy(genotype_container)
            new_node.genome.mutate_add_node(self.config.genome_config)
            
            generated_genotypes.add(new_node)
            self.connections.append(Connection(genotype_container, new_node))
            GenotypePhenotypeMap.id_counter += 1
            new_node.id = GenotypePhenotypeMap.id_counter
            
            mutations_applied.add(genotype_container)
                
        # Iterates through generated phenotypes, producing their associated phenotypes
        for genotype in generated_genotypes:
            net = neat.nn.FeedForwardNetwork.create(genotype.genome, self.config) # Network used to create xenobot bodies
            body = genotype_to_phenotype(net, [8,8,7]) # Creates xenobot body using neural network
            body = tuple(body) # Makes body hashable
            
            # Adds phenotype to genotype-phenotype mapping
            if body in self.map:
                self.map[body].append(genotype)
            else:
                self.map[body] = [genotype]  
    
    def num_genotypes_and_phenotypes(self) -> Tuple:
        """
        Function to get the total number of genotypes
        and total number of phenotypes
        """
        genotypes = 0
        phenotypes = len(self.map.keys())
        for key in self.map:
            for _ in self.map[key]:
                genotypes += 1
        
        return (genotypes, phenotypes)
    
    def gen_phenotype_info(self):
        """ 
        Function to generate information about phenotypes
        within the genotype_phenotype map
        """
        self.phenotypes = []
        for phenotype in self.map:
            complexity = calc_KC(str(phenotype))
            num_mapped_genotypes = len(self.map[phenotype])
            container = Phenotype(phenotype, complexity, num_mapped_genotypes)
            self.phenotypes.append(container)
            for genotype in self.map[phenotype]:
                genotype.phenotype = container

    def most_designable(self) -> List:
        """ 
        Returns a list of the most designable phenotypes,
        meaning the phenotype with the most amount of genotypes
        which map to it
        """
        sorted_phenotypes = sorted(self.phenotypes, key=lambda x: x.genotypes, reverse=True)
        return sorted_phenotypes

    def most_complex(self) -> List:
        """
        Returns a sorted list of phenotypes by their complexity,
        as calculated via the LZW algorithm
        """
        sorted_phenotypes = sorted(self.phenotypes, key=lambda x: x.complexity, reverse=True)
        return sorted_phenotypes

    def mean_genotype_to_phenotype_ratio(self) -> float:
        """ 
        Calculate the mean ratio of genotypes to phenotypes
        """
        genotypes, phenotypes = self.num_genotypes_and_phenotypes()
        return genotypes/phenotypes

    def random_neutral_walk(self, start_genome, steps) -> List:
        """ 
        Random neutral walk through the genotype space of the 
        genotype-phenotype map.
        Walks through genotype space, only moving to the next genotype
        if its phenotype matches that of the current one.
        
        :param start_genome: 
        :param steps: 
        :rtype: List
        :return: List of genotypes traversed in the walk path
        """
        walk_path = [start_genome]
        current_genome = start_genome
        encountered_phenotypes = set() # Phenotypes encountered during walk
        
        for _ in range(steps):
            # Gets a list of the valid traversals
            valid_traversals = []
            for connection in self.connections:
                if connection.n1 is current_genome:
                    valid_traversals.append(connection.n2)
                elif connection.n2 is current_genome:
                    valid_traversals.append(connection.n1)
                    
            if len(valid_traversals) > 0:
                next_genome = choice(valid_traversals) # Chooses next genotype
                
                # Checks if phenotypes match
                if np.array_equal(next_genome.phenotype.phenotype, current_genome.phenotype.phenotype):
                    walk_path.append(next_genome) # Adds next genotype to walk path
                    current_genome = next_genome # Moves walk to next node
                else:
                    encountered_phenotypes.add(tuple(next_genome.phenotype.phenotype))
                
        return walk_path, encountered_phenotypes
    
    def probability_of_phenotypes(self) -> List:
        """ 
        Function to calculate the probability of encountering each phenotype.
        This identifies if the evolutionary system has a bias in producing
        a given phenotype output.
        
        :rtype: List
        :return: List of probabilities of encountering each phenotype in the system
        """
        total_genotypes = 0
        phenotype_probabilities = []
        for phenotype in self.map:
            total_genotypes += len(self.map[phenotype])
            phenotype_probabilities.append(len(self.map[phenotype]))
        
        return [x / total_genotypes for x in phenotype_probabilities]
    
    def probability_of_complex_phenotypes(self) -> List:
        """ 
        Function to calculate the probability of encountering phenotypes of 
        varying complexities. This identifies if the evolutionary system has a
        bias in producing phenotypes of a given complexity.
        
        :rtype: List
        :return: List of probabilities of encountering complexitie of each phenotype in the system
        """
        total_genotypes = 0
        phenotype_probabilities = np.zeros(10) # Empty phenotype probabilities array
        for phenotype in self.map:
            total_genotypes += len(self.map[phenotype]) # Increments total number of
            for genotype in self.map[phenotype]:
                complexity = round(genotype.phenotype.complexity)
                str_complexity = str(complexity)
                if len(str_complexity) == 2:
                    phenotype_probabilities[0] += len(self.map[phenotype])
                else:
                    phenotype_probabilities[int(str_complexity[0])] += len(self.map[phenotype])
                    
        return [x / total_genotypes for x in phenotype_probabilities]
                
    def reachability(self) -> List:
        """ 
        
        """
        # Can make histograms using this!!
        num_phenotypes_encountered = []

        for phenotype in self.map:
            for _ in range(100):
                random_genotype = choice(self.map[phenotype])
                _, phenotypes_encountered = self.random_neutral_walk(random_genotype, 300)
                num_phenotypes_encountered.append(len(phenotypes_encountered))
        
        return num_phenotypes_encountered
    
    def innovation_rate(self) -> float:
        """ 
        
        """
        #TODO
        for phenotype in self.map:
            for _ in range(100):
                random_genotype = choice(self.map[phenotype])
                self.random_neutral_walk(random_genotype, 100)
    
    def calculate_fitness(self) -> None:
        """ 
        Calculates and assigns
        
        TODO: Fitness for range of applications using vxa.set_fitness_function
        """
        fitness_categories = ["abs_movement", ""]
        
        fitness_file_mapping = {}
        
        vxa = VXA(SimTime=3, HeapSize=0.65, RecordStepSize=100, DtFrac=0.95, EnableExpansion=1)
        passive = vxa.add_material(RGBA=(0,255,0), E=5000000, RHO=1000000) # passive soft
        active = vxa.add_material(RGBA=(255,0,0), CTE=0.01, E=5000000, RHO=1000000) # active
        
        os.system(f"rm -rf gp-fitness/") #Deletes contents of run directory if exists
        os.system(f"mkdir -p gp-fitness") # Creates a new directory to store fitness files
        vxa.write("base.vxa") #Write a base vxa file
        os.system(f"cp base.vxa gp-fitness") #Copy vxa file to correct run directory
        os.system("rm base.vxa") #Removes old vxa file
        
        #TODO EVALUATE FITNESS IN BATCHES OF 100 OTHERWISE DOES NOT WORK
        
        evaluated = 0
    
        for phen_index, phenotype in enumerate(self.map):
            evaluated += 1
            fitness_file_mapping[phen_index] = phenotype
            
            vxd = VXD()
            vxd.set_tags(RecordVoxel=1)
            body = np.zeros(8*8*7)
            
            for cell in range(len(phenotype)):
                if phenotype[cell] == 1:
                    body[cell] = passive
                elif phenotype[cell] == 2:
                    body[cell] = active 
            
            reshaped = body.reshape(8,8,7)
            vxd.set_data(reshaped)
            
            vxd.write(f"id{phen_index}.vxd") #Writes vxd file for individual
            os.system(f"cp id{phen_index}.vxd gp-fitness/")
            os.system(f"rm id{phen_index}.vxd") #Removes the old non-copied vxd file
            
        
        os.chdir("voxcraft-sim/build") # Changes directory to the voxcraft directory TODO change to be taken from settings file
        os.system(f"./voxcraft-sim -i ../../gp-fitness -o ../../gp-fitness/output.xml -f > ../../gp-fitness/test.history")
        os.chdir("../../") # Return to project directory
        
        results = read_sim_output(f"gp-fitness/output") #Reads sim results from output file
        os.system("rm -rf gp-fitness")
        
        for result in results:
            phenotype_index = result["index"]
            
            #TODO Verify this works
            for genotype in self.map[fitness_file_mapping[phenotype_index]]:
                genotype.phenotype.fitness["abs_movement"] = float(result["fitness"])

class MultiLayeredGenotypePhenotypeMap(GenotypePhenotypeMap):
    """ 
    
    """
    def __init__(self) -> None:
        self.map = {}
        self.cppn_to_nn_map = {}

def save(
    obj, 
    filename: str
    ) -> None:
    """ 
    Method to pickle a genotype-phenotype map
    
    :param filename: name of the file to save object to
    """
    with open(filename, "wb") as file:
        pickle.dump(obj, file)

def load(filename: str) -> None:
    """ 
    Method to load a gentype-phenotype map
    
    :param filename: name of the file from which to load gp map
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
        return obj


if __name__ == "__main__":
    # gp = GenotypePhenotypeMap("config-gpmap")
    
    # gp.initilise(1000)
    
    # save(gp, "genotype-phenotype maps/3000 genotypes")
    
    gp = load("genotype-phenotype maps/3000 genotypes")
    
    for phenotype in gp.map:
        pheno = gp.map[phenotype][0].phenotype
        print(pheno.fitness)
