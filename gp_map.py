""" 

"""
# %%
import pickle
import neat
from copy import deepcopy
from random import choice
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt
from tools.activation_functions import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs, normalize
from tools.gen_phenotype_from_genotype import genotype_to_phenotype
from visualise_xenobot import show

class Geneotype:
    """ 
    
    """
    def __init__(
        self, 
        geneome
        ) -> None:
        self.genome = geneome
        self.previous = None

class GenotypePhenotypeMap:
    """ 
    Class representing a genotype-phenotype map of 
    CPPNs and the xenobots created by each
    """
    def __init__(
        self, 
        config_name: str
        ) -> None:
        self.map = {}
        self.phenotype_fitness = {}
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_name)
        self.config.genome_config.add_activation("neg_abs", neg_abs)
        self.config.genome_config.add_activation("neg_square", neg_square)
        self.config.genome_config.add_activation("sqrt_abs", sqrt_abs)
        self.config.genome_config.add_activation("neg_sqrt_abs", neg_sqrt_abs)
    
    def gen_genotypes_of_n(
        self, 
        n: int
        ) -> None:
        """
        Function to generate genotypes... 
        
        :param n: size of genotypes to generate (number of nodes)
        """    
        generated_genotypes = set()
        mutations_applied = set()
        START_SIZE = 7
        
        default_genome = neat.DefaultGenome(1)
        default_genome.configure_new(self.config.genome_config)
        
        genotype_container = Geneotype(default_genome)
        generated_genotypes.add(genotype_container)
        
        for _ in range(100):
            not_explored = generated_genotypes - mutations_applied
            genotype_container = choice(tuple(not_explored))
        
            # TODO CHECK THESE MUTATIONS AND HAVE MORE CONTROL OVER THEM
            # DISCRETIZE THEM
            # Connection weights
            new_connection_weights = deepcopy(genotype_container)
            for connection in new_connection_weights.genome.connections.values():
                connection.mutate(self.config.genome_config)
        
            generated_genotypes.add(new_connection_weights)
        
            # Node activation functions and bias
            new_activation_functions = deepcopy(genotype_container)
            for node in new_activation_functions.genome.nodes.values():
                node.mutate(self.config.genome_config)
        
            generated_genotypes.add(new_activation_functions)
        
            # Add connections
            new_connections = deepcopy(genotype_container)
            new_connections.genome.mutate_add_connection(self.config.genome_config)
        
            generated_genotypes.add(new_connections)
        
            mutations_applied.add(genotype_container)
        
        for genotype in generated_genotypes:
            net = neat.nn.FeedForwardNetwork.create(genotype.genome, self.config)
            body = genotype_to_phenotype(net, [8,8,7])
            show(body)
                
            
    def draw(self):
        """ 
        Draws the genotype-phenotype map using networkx
        """
        # Create pandas dataframe
        graph = nx.Graph()
        graph.add_node(1)
        
        nx.draw_networkx(graph)
        

class MultiLayeredGenotypePhenotypeMap:
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
    """
    with open(filename, "wb") as file:
        pickle.dump(obj, file)

def load(filename: str):
    """ 
    Method to load a gentype-phenotype map
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
        return obj


gp = GenotypePhenotypeMap("config-gpmap")

gp.gen_genotypes_of_n(10)



