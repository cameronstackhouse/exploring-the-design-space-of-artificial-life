import pickle
from cppn_neat import mutate_connection, mutate_node, remove_connection, add_connection, add_node_between_con
from networks import CPPN
from graphviz import Digraph
#TODO Use graphviz

class GenotypePhenotypeMap:
    """
    Class to represent a genotype-phenotype map
    """

    def __init__(self, num_genotypes) -> None:
        """
        
        """
        self.gp_map = {}
        #NOTE Probably mapped phenotype: [genotypes]

        #TODO Generate Genotypes
        genotypes = 0
        initial_genotype = CPPN([8,8,7]) #Creates a random starting genotype
        current = initial_genotype
        while genotypes < num_genotypes:
            #6 Potential genotypes to be generated from different mutations
            if len(current.nodes) > 2:
                #TODO CAN REMOVE NODE!
                pass

        #TODO Map neutral network

        #TODO Map each Genotype to Phenotype


class FitnessLandscape:
    """
    Class to represent a fitness landscape, including a genotype-phenotype map
    and a mapping of phenotype to fitness
    """
    def __init__(self, num_genotypes) -> None:
        self.gp_map = GenotypePhenotypeMap(num_genotypes) #Generates genotype-phenotype map
        self.phenotype_fitness_map = {} #NOTE Probably mapped fitness: [phenotypes] #Generates phenotype-fitness map

        for key in self.gp_map:
            phenotype_fitness = self.gp_map[key][0].fitness

            if phenotype_fitness in self.phenotype_fitness_map.keys():
                self.phenotype_fitness_map[phenotype_fitness].append(key)
            else:
                self.phenotype_fitness_map[phenotype_fitness] = [key]
    
    def visualize(self) -> None:
        pass
