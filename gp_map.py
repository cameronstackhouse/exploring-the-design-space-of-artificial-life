""" 
Module implementing the ability to generate genotype-phenotype maps
"""
# %%
import pickle
import neat
from copy import deepcopy
from random import choice
from typing import Tuple, List
import networkx as nx 
from tools.activation_functions import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs, normalize
from tools.gen_phenotype_from_genotype import genotype_to_phenotype
from tools.phenotype_information import calc_KC
from visualise_xenobot import show

class Genotype:
    """ 
    Class to represent a genotype in the genotype map.
    parent_of attribute essential for visualisation of 
    GP map
    """
    def __init__(
        self, 
        geneome,
        id
        ) -> None:
        self.genome = geneome
        self.parent_of = []
        self.id = id

class Phenotype:
    """ 
    
    """
    def __init__(
        self,
        phenotype,
        complexity,
        genotypes
        ) -> None:
        self.phenotype = phenotype
        self.fitness = {} # Dictionary for fitnesses at 
        self.complexity = complexity
        self.genotypes = genotypes

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
        self.graph = None
        self.phenotypes = []
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
        
        genotype_container = Genotype(default_genome, self.id_counter)
        generated_genotypes.add(genotype_container)
        
        for _ in range(10):
            not_explored = generated_genotypes - mutations_applied
            genotype_container = choice(tuple(not_explored))
        
            # TODO CHECK THESE MUTATIONS AND HAVE MORE CONTROL OVER THEM
            # DISCRETIZE THEM
            # Connection weights
            new_connection_weights = deepcopy(genotype_container)
            for connection in new_connection_weights.genome.connections.values():
                connection.mutate(self.config.genome_config)
        
            generated_genotypes.add(new_connection_weights)
            genotype_container.parent_of.append(new_connection_weights)
            GenotypePhenotypeMap.id_counter += 1
            new_connection_weights.id = GenotypePhenotypeMap.id_counter 
            
            # Node activation functions and bias
            new_activation_functions = deepcopy(genotype_container)
            for node in new_activation_functions.genome.nodes.values():
                node.mutate(self.config.genome_config)
        
            generated_genotypes.add(new_activation_functions)
            genotype_container.parent_of.append(new_activation_functions)
            GenotypePhenotypeMap.id_counter += 1
            new_activation_functions.id = GenotypePhenotypeMap.id_counter 
            
        
            # Add connections
            new_connections = deepcopy(genotype_container)
            new_connections.genome.mutate_add_connection(self.config.genome_config)
        
            generated_genotypes.add(new_connections)
            genotype_container.parent_of.append(new_connections)
            GenotypePhenotypeMap.id_counter += 1
            new_connections.id = GenotypePhenotypeMap.id_counter 
        
            mutations_applied.add(genotype_container)
        
        # for genotype in generated_genotypes:
        #     net = neat.nn.FeedForwardNetwork.create(genotype.genome, self.config)
        #     body = genotype_to_phenotype(net, [8,8,7])
        #     show(body)
        
        for genotype in generated_genotypes:
            net = neat.nn.FeedForwardNetwork.create(genotype.genome, self.config)
            body = genotype_to_phenotype(net, [8,8,7])
            body = tuple(body)
            
            if body in self.map:
                self.map[body].append(genotype)
            else:
                self.map[body] = [genotype]  
        
        for p in self.map:
            assert self.map[p] is not None
    
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
        
        """
        self.phenotypes = []
        for phenotype in self.map:
            complexity = calc_KC(str(phenotype))
            num_mapped_genotypes = len(self.map[phenotype])
            container = Phenotype(phenotype, complexity, num_mapped_genotypes)
            self.phenotypes.append(container)

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
        
        """
        sorted_phenotypes = sorted(self.phenotypes, key=lambda x: x.complexity, reverse=True)
        return sorted_phenotypes

    def mean_genotype_to_phenotype_ratio(self) -> float:
        """ 
        Calculate the mean ratio of genotypes to phenotypes
        """
        genotypes, phenotypes = self.num_genotypes_and_phenotypes()
        return genotypes/phenotypes
              
    def create_graph(self) -> None:
        """ 
        Function to create a networkx graph representing the
        genotype-phenotype mapping 
        """
        graph = nx.Graph()
        
        # Genotype Nodes
        # Adds nodes to graph
        genotypes, _ = self.num_genotypes_and_phenotypes()
        for n in range(genotypes):
            graph.add_node(n)
            
        # Adds connections to graph
        for phenotype in self.map:
            for genotype in self.map[phenotype]:
                for connected_to in genotype.parent_of:
                    graph.add_edge(genotype.id, connected_to.id)
        
        # Add Phenotype Information
        # TODO Connections between phenotypes somehow!
        phenotype_id_counter = GenotypePhenotypeMap.id_counter + 1
        phenotype_with_id = {}
        for phenotype in self.map:
            graph.add_node(phenotype_id_counter)
            phenotype_with_id[phenotype] = phenotype_id_counter
            phenotype_id_counter += 1
        
        for phenotype in self.map:
            for genotype in self.map[phenotype]:
                graph.add_edge(phenotype_with_id[phenotype], genotype.id)
        
        self.graph = graph
            
    def draw(self):
        """ 
        Draws the genotype-phenotype map using networkx
        """
        # Create pandas dataframe
        self.create_graph()
        
        colour_map = ['red' if node <= GenotypePhenotypeMap.id_counter else 'green' for node in self.graph]
        
        nx.draw_spring(self.graph, node_color=colour_map)
        

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


gp = GenotypePhenotypeMap("config-gpmap")

gp.gen_genotypes_of_n(10)

ratio = gp.mean_genotype_to_phenotype_ratio()

print(f"Genotype -> Phenotype ratio: {ratio}")

gp.draw()

gp.gen_phenotype_info()

a = gp.most_designable()

for phenotype in a:
    print(f"Number of genotypes: {phenotype.genotypes}. Complexity: {phenotype.complexity}")