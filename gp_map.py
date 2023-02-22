""" 

"""

  #%%

import pickle
import neat
import networkx as nx 
import matplotlib.pyplot as plt
from tools.activation_functions import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs

class Geneotype:
    """ 
    
    """
    def __init__(
        self, 
        geneome
        ) -> None:
        self.genome = geneome
        self.previous = []

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
        default_genome = neat.DefaultGenome(1)
        default_genome.configure_new(self.config.genome_config)
        
        #start_net = neat.nn.FeedForwardNetwork.create(default_genome, config)
        
        #Apply mutations
        
        # TODO CHECK THESE MUTATIONS AND HAVE MORE CONTROL OVER THEM
        # Connection weights
        for connection in default_genome.connections.values():
            connection.mutate(self.config.genome_config)
        
        # Node activation functions and bias
        for node in default_genome.nodes.values():
            node.mutate(self.config.genome_config)
        
        # Add connections
        default_genome.mutate_add_connection(self.config.genome_config)
        
        
        print(default_genome.size())
            
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
    
    """
    with open(filename, "wb") as file:
        pickle.dump(obj, file)

def load(filename: str):
    """ 
    
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
        return obj


gp = GenotypePhenotypeMap("config-gpmap")

gp.gen_genotypes_of_n(10)

gp.draw()

