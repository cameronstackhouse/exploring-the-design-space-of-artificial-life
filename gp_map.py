""" 
Module implementing the ability to generate genotype-phenotype maps
"""
# %%
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
from visualise_xenobot import show

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
    
    """
    def __init__(self, n1, n2) -> None:
        """ 
        
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
        self.map = {}
        self.phenotypes = []
        self.connections = []
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_name)
        self.config.genome_config.add_activation("neg_abs", neg_abs)
        self.config.genome_config.add_activation("neg_square", neg_square)
        self.config.genome_config.add_activation("sqrt_abs", sqrt_abs)
        self.config.genome_config.add_activation("neg_sqrt_abs", neg_sqrt_abs)
    
    def gen_general_phenotype_map(self) -> None:
        pass
    
    def gen_genotypes_of_n(
        self, 
        n: int
        ) -> None:
        """
        Function to generate genotypes... 
        
        :param n: size of genotypes to generate (number of nodes)
        """    
        
        # TODO Change to bidirectional connection system
        generated_genotypes = set() # Set of generated genotypes
        mutations_applied = set() # Set of genotypes which mutations have been applied to
        START_SIZE = 7
        CONNECTION_WEIGHTS = np.arange(-2, 2, 0.25) # Range of possible connection weights
        ACTIVATION_FUNCTIONS = ["sigmoid", "sin", "neg_abs", "square", "neg_square", "sqrt_abs", "neg_sqrt_abs"]
        
        default_genome = neat.DefaultGenome(1) # Creates a default genome
        default_genome.configure_new(self.config.genome_config) # Configures this default genome
        
        genotype_container = Genotype(default_genome, self.id_counter) # Adds genome to genotype container
        generated_genotypes.add(genotype_container) # Adds the initial genome to the set of generated genotypes
        
        for n in range(1000):
            print(n)
            # Set of genotypes which havent had mutations applied to them yet
            not_explored = generated_genotypes - mutations_applied
            genotype_container = choice(tuple(not_explored)) # Chooses genotypem from set to apply set of mutations to
        
            # Applies possible range of mutations to the chosen genotype to generate neighbours
            
            # Connection weights
            new_connection_weights = deepcopy(genotype_container) # Creates a copy of the genotype
            # Mutates the connection weights in the 
            for connection in new_connection_weights.genome.connections.values():
                #TODO: Discretize this!!
                # Update connection.weight int
                connection.mutate(self.config.genome_config)
        
            generated_genotypes.add(new_connection_weights)
            self.connections.append(Connection(genotype_container, new_connection_weights))
            
            GenotypePhenotypeMap.id_counter += 1
            new_connection_weights.id = GenotypePhenotypeMap.id_counter 
            
            # Node activation functions and bias
            # Node.activation 'str'
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
                
        for genotype in generated_genotypes:
            net = neat.nn.FeedForwardNetwork.create(genotype.genome, self.config)
            body = genotype_to_phenotype(net, [8,8,7])
            body = tuple(body)
            
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
        :rtypr: List
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
        
        """
        total_genotypes = 0
        phenotype_probabilities = []
        for phenotype in self.map:
            total_genotypes += len(self.map[phenotype])
            phenotype_probabilities.append(len(self.map[phenotype]))
        
        return [x / total_genotypes for x in phenotype_probabilities]
    
    def probability_of_complex_phenotypes(self) -> List:
        """ 
        
        """
        pass
    
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

class MultiLayeredGenotypePhenotypeMap(GenotypePhenotypeMap):
    """ 
    
    """
    def __init__(self) -> None:
        self.map = {}

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


#gp = GenotypePhenotypeMap("config-gpmap")

#p.gen_genotypes_of_n(10)

gp = load("temp")

ratio = gp.mean_genotype_to_phenotype_ratio()

print(f"Genotype -> Phenotype ratio: {ratio}")

#gp.gen_phenotype_info()

probs = gp.probability_of_phenotypes()

sns.histplot(gp.reachability())