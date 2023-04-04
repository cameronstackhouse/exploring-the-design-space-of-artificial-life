""" 
Module implementing the ability to generate genotype-phenotype maps
"""
# %%``
import os
import pickle
import neat
import json
from math import log10
from copy import deepcopy
from random import choice
from typing import Tuple, List
from collections import defaultdict
import numpy as np
from tools.activation_functions import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs
from tools.gen_phenotype_from_genotype import genotype_to_phenotype
from tools.phenotype_information import calc_KC
from tools.read_files import read_sim_output
from visualise_xenobot import show
from voxcraftpython.VoxcraftVXD import VXD
from voxcraftpython.VoxcraftVXA import VXA
from pureples.shared.substrate import Substrate
from pureples.es_hyperneat.es_hyperneat import ESNetwork
import seaborn as sns
import matplotlib.pyplot as plt

# TODO:MODIFY FUNCTIONS FOR HYPERNEAT GP MAP

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
        self.label = None

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
        self.num_cppn_designers = None # TODO ENSURE THIS IS DONE FOR HYPERNEAT MODIFY ALL FUNCTIONS
        self.motifs = None

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
        config_name: str,
        hyperneat: bool = False
        ) -> None:
        """ 
        Initilises a genotype-phenotype map object given 
        """
        self.cppn_to_gene = {} # Used for hyperneat
        self.map = {}
        self.phenotypes = []
        self.connections = []
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_name)
        self.substrate = None
        self.params = None
        self.hyperneat = hyperneat
        
        self.config.genome_config.add_activation("neg_abs", neg_abs)
        self.config.genome_config.add_activation("neg_square", neg_square)
        self.config.genome_config.add_activation("sqrt_abs", sqrt_abs)
        self.config.genome_config.add_activation("neg_sqrt_abs", neg_sqrt_abs)

        # Set substrate, set params
        self.params = {"initial_depth": 2,
                        "max_depth": 3,
                        "variance_threshold": 0.03,
                        "band_threshold": 0.3,
                        "iteration_level": 1,
                        "division_threshold": 0.5,
                        "max_weight": 5.0,
                        "activation": "sigmoid"}
            
        INPUT_COORDINATES = []
            
        for i in range(0, 5):
            INPUT_COORDINATES.append((-1 + (2 * i/3), -1))
                        
        OUTPUT_COORDINATES = [(-1, 1), (1, 1)]
            
        self.substrate = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)
    
    def initilise(self, num_genotypes: int) -> None:
        """ 
        
        """
        self.gen_genotypes(num_genotypes)
        self.gen_phenotype_info()
        #self.calculate_fitness()
        
    def gen_genotypes(
        self, 
        num_genotypes: int
        ) -> None:
        """
        Function to generate genotypes... 
        TODO 
        
        :param n: size of genotypes to generate (number of nodes)
        """    
        generated_genotypes = set() # Set of generated genotypes
        mutations_applied = set() # Set of genotypes which mutations have been applied to
        
        default_genome = neat.DefaultGenome(1) # Creates a default genome
        default_genome.configure_new(self.config.genome_config) # Configures this default genome
        
        genotype_container = Genotype(default_genome, self.id_counter) # Adds genome to genotype container
        generated_genotypes.add(genotype_container) # Adds the initial genome to the set of generated genotypes
        
        number = 0
        
        while number <= num_genotypes:
            # Set of genotypes which havent had mutations applied to them yet
            not_explored = generated_genotypes - mutations_applied
            genotype_container = choice(tuple(not_explored)) # Chooses genotypem from set to apply set of mutations to
        
            # Applies possible range of mutations to the chosen genotype to generate neighbours
            
            # Applies mutation to every connection
            for n, _ in enumerate(genotype_container.genome.connections.values()):
                new_connection_weights = deepcopy(genotype_container) # Creates a copy of the genotype
                list(new_connection_weights.genome.connections.values())[n].mutate(self.config.genome_config)
                generated_genotypes.add(new_connection_weights)
                self.connections.append(Connection(genotype_container, new_connection_weights))
                GenotypePhenotypeMap.id_counter += 1
                new_connection_weights.id = GenotypePhenotypeMap.id_counter 
                number+=1
            
            # Mutates activation function of each node
            for n, _ in enumerate(genotype_container.genome.nodes.values()):
                new_activation_function = deepcopy(genotype_container)
                list(new_activation_function.genome.nodes.values())[n].mutate(self.config.genome_config)
                generated_genotypes.add(new_activation_function)
                self.connections.append(Connection(genotype_container, new_activation_function))
                GenotypePhenotypeMap.id_counter += 1
                new_activation_function.id = GenotypePhenotypeMap.id_counter 
                number+=1
            
            # Add connection
            new_connections = deepcopy(genotype_container)
            new_connections.genome.mutate_add_connection(self.config.genome_config)
            generated_genotypes.add(new_connections)
            self.connections.append(Connection(genotype_container, new_connections))
            GenotypePhenotypeMap.id_counter += 1
            new_connections.id = GenotypePhenotypeMap.id_counter 
            number+=1
            
            # Add node
            new_node = deepcopy(genotype_container)
            new_node.genome.mutate_add_node(self.config.genome_config)
            generated_genotypes.add(new_node)
            self.connections.append(Connection(genotype_container, new_node))
            GenotypePhenotypeMap.id_counter += 1
            new_node.id = GenotypePhenotypeMap.id_counter
            number+=1
            
            mutations_applied.add(genotype_container)
                
        # Iterates through generated phenotypes, producing their associated phenotypes
        for genotype in generated_genotypes:
            net = None
            if self.hyperneat:
                cppn = neat.nn.FeedForwardNetwork.create(genotype.genome, self.config) # CPPN to design network to produce xenobot
                sub = ESNetwork(self.substrate, cppn, self.params) # created substrate
                
                # Adds mapping between cppn and network to design xenobots
                if sub in self.cppn_to_gene:
                    self.cppn_to_gene[sub].append(cppn)
                else:
                    self.cppn_to_gene[sub] = [cppn]
                
                net = sub.create_phenotype_network()
            else:
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
        
        if self.hyperneat:
            producer_cppns = 0
            #TODO MAKE SURE THIS WORKS
            for key in self.cppn_to_gene:
                for _ in self.cppn_to_gene[key]:
                    producer_cppns += 1
                
            return (producer_cppns, genotypes, phenotypes)
        
        return (genotypes, phenotypes)
    
    def gen_phenotype_info(self):
        """ 
        Function to generate information about phenotypes
        within the genotype_phenotype map
        """
        #TODO ENSURE THIS WORKS WITH HYPERNEAT
        self.phenotypes = []
        for phenotype in self.map:
            pheno_string = ""
            for cell in phenotype:
                pheno_string += str(int(cell))
            complexity = calc_KC(pheno_string)
            num_mapped_genotypes = len(self.map[phenotype])
            container = Phenotype(phenotype, complexity, num_mapped_genotypes)
            self.phenotypes.append(container) # Adds phenotype object
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
        if self.hyperneat:
            for phenotype in self.phenotypes:
                total_genotypes += phenotype.num_cppn_designers
                phenotype_probabilities.append(phenotype.num_cppn_designers)
        else:
            for phenotype in self.map:
                total_genotypes += len(self.map[phenotype])
                phenotype_probabilities.append(len(self.map[phenotype]))
        
        return [x / total_genotypes for x in phenotype_probabilities]
    
    def calculate_fitness(self) -> None:
        """ 
        Calculates and assigns
        
        TODO: Fitness for range of applications using vxa.set_fitness_function
        """
        # TODO, MORE FITNESS CATEGORIES
        #fitness_categories = ["abs_movement", ""]
        
        fitness_file_mapping = {}
        
        vxa = VXA(SimTime=3, HeapSize=0.65, RecordStepSize=100, DtFrac=0.95, EnableExpansion=1)
        passive = vxa.add_material(RGBA=(0,255,0), E=5000000, RHO=1000000) # passive soft
        active = vxa.add_material(RGBA=(255,0,0), CTE=0.01, E=5000000, RHO=1000000) # active
        
        os.system(f"rm -rf gp-fitness/") #Deletes contents of run directory if exists
        os.system(f"mkdir -p gp-fitness") # Creates a new directory to store fitness files
        vxa.write("base.vxa") #Write a base vxa file
        os.system(f"cp base.vxa gp-fitness") #Copy vxa file to correct run directory
        os.system("rm base.vxa") #Removes old vxa file
                
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
            
            #TODO CHECK IF BODY ALL 0s or 1s: NO FITNESS
            
            reshaped = body.reshape(8,8,7)
            vxd.set_data(reshaped)
            
            vxd.write(f"id{phen_index}.vxd") #Writes vxd file for individual
            os.system(f"cp id{phen_index}.vxd gp-fitness/")
            os.system(f"rm id{phen_index}.vxd") #Removes the old non-copied vxd file
            
            if (evaluated >= 100) or (phen_index == len(self.map) - 1):
                os.chdir("voxcraft-sim/build") # Changes directory to the voxcraft directory TODO change to be taken from settings file
                os.system(f"./voxcraft-sim -i ../../gp-fitness -o ../../gp-fitness/output.xml -f > ../../gp-fitness/test.history")
                os.chdir("../../") # Return to project directory
        
                results = read_sim_output(f"gp-fitness/output") #Reads sim results from output file
        
                os.system("rm -rf gp-fitness")
                os.system(f"mkdir -p gp-fitness") # Creates a new directory to store fitness files
                vxa.write("base.vxa") #Write a base vxa file
                os.system(f"cp base.vxa gp-fitness") #Copy vxa file to correct run directory
                os.system("rm base.vxa") #Removes old vxa file
        
                for result in results:
                    phenotype_index = result["index"]
            
                    #Assign absolute moevement fitness value to phenotype
                    for genotype in self.map[fitness_file_mapping[phenotype_index]]:
                        genotype.phenotype.fitness["abs_movement"] = float(result["fitness"])
                
                evaluated = 0
        
        os.system("rm -rf gp-fitness")
                                            

def gen_motifs(phenotypes: list) -> set:
    """ 
    Generates all 3x3x3 structural 
    motifs present in xenobots in the GP map.
            
    :rtype: set
    :return: set of motifs identified in the population of xenobots
    """
    #TODO Change to any sized motifs
    
    # TODO: CHANGE TO ONLY KEEPING COUNTS OF 1000+
    
    phenotype_counts = defaultdict(int)
    motifs = set() # Set of unique motifs
    for phenotype in phenotypes:
        body = np.array(phenotype.phenotype)
        shaped = body.reshape(8,8,7)
        # Sliding window identifying motifs of size 3x3x3
        for i in range(len(shaped) - 2):
            for j in range(len(shaped[0]) - 2):
                for k in range(len(shaped[0][0]) - 2):
                    motif = shaped[i:i+3, j:j+3, k:k+3]
                    hashed = hash(tuple(motif.flatten()))
                    phenotype_counts[hashed] += 1
                    
                    if phenotype_counts[hashed] >= 1000:
                        motifs.add(tuple(motif.flatten()))
          
    return motifs

def count_motifs(
    phenotypes: list, 
    motifs: set,
    motif_index: dict
    )-> None:
    """ 
    Counts the number of 4x4x4 motifs in each phenotype 
    given a list of motifs.
    
    Only saves motifs 
            
    :param motifs: iterable collection of motifs to count
    """
    #TODO Change to any size sized motifs
    
    count = 0
    for phenotype in phenotypes:
        motif_counts = defaultdict(int)
        body = np.array(phenotype.phenotype)
        shaped = body.reshape(8,8,7)
        # Sliding window to identify subsections of xenobot
        for i in range(len(shaped) - 2):
            for j in range(len(shaped[0]) - 2):
                for k in range(len(shaped[0][0]) - 2):
                    subsection = shaped[i:i+3, j:j+3, k:k+3]
                    indexable = tuple(subsection.flatten())
                    if indexable in motifs: # Checks if indexed subsection is in list of motifs
                        motif_counts[motif_index[indexable]] += 1
        
        count += 1
        print(f"Calculated motifs for phenotype {count} out of {len(phenotypes)}")
        phenotype.motifs = motif_counts # Sets phenotype motifs

def gen_motif_clustering_data_file(phenotypes, motif_name) -> None:
    """ 
    
    """
    motifs = set(load(motif_name))
    print("MOTIFS LOADED")
    motif_index = {motif: i for i, motif in enumerate(motifs)}
    count_motifs(phenotypes, motifs, motif_index)
    print("COUNTED")
    
    motif_index_string = {}
    
    for n, motif in enumerate(motifs):
        motif_string = ""
        for cell in motif:
            motif_string += str(int(cell))
        motif_index_string[n] = motif_string
    
    data = {
        "motifs": motif_index_string,
        "xenobots":[]
    }
    
    for phenotype in phenotypes:
        xenobot_entry = {}
        
        str_body = ""
        for cell in phenotype.phenotype:
            str_body += str(int(cell))
        
        xenobot_entry["body"] = str_body
        xenobot_entry["motif_counts"] = phenotype.motifs
        
        data["xenobots"].append(xenobot_entry)
    
    json_object = json.dumps(data, indent=4)
    
    with open("motif_data.json", "w") as outfile:
        outfile.write(json_object)
        
def complexity_probability_distribution(gp_map: GenotypePhenotypeMap):
    """ 
    
    """
    #TODO add comments
    count = []
    complexity = []
    total = 0
    for i in range(len(gp_map.phenotypes)):
        count.append(gp_map.phenotypes[i].genotypes)
        total += count[i]
        complexity.append(gp_map.phenotypes[i].complexity)
    
    count = [log10((c/total)) for c in count]
    plt.scatter(complexity, count, s=1)
    plt.ylabel("$Log_{10}(P)$")
    plt.xlabel("Estimated Kolmogorov complexity")

def robustness(gp_map) -> float:
    """ 
    Calculates the mean genotypic robustness of a genotype-phenotype mapping,
    meaning the mean number of genotype mutations which do not 
    alter the resulting phenotype structure.
    """
    #TODO calculate
    
    # Gets genotype connections
    connections = defaultdict(list)
    
    for connection in gp_map.connections:
        connections[connection.n1].append(connection.n2)
        connections[connection.n2].append(connection.n1)
    
    counter = 0
    total = 0
    for phenotype in gp_map.map:
        for genotype in gp_map.map[phenotype]:
            
            if len(connections[genotype]) > 1:
                counter += 1
                matching_phenotype = 0
                num_neighbours = len(connections[genotype])
                for connected_to in connections[genotype]:
                    if connected_to.phenotype.phenotype == genotype.phenotype.phenotype:
                        matching_phenotype += 1
                
                total += (matching_phenotype/num_neighbours)
    
    return total / counter

def phenotypic_robustness(gp_map) -> float:
    """ 
    Average of genotypic robustness over genotypes that match to that phenotype
    """
    connections = defaultdict(list)
    
    for connection in gp_map.connections:
        connections[connection.n1].append(connection.n2)
        connections[connection.n2].append(connection.n1)
    
    total = 0
    num_calculated = 0
    
    for pheno in gp_map.map:
        pheno_robustness = 0
        counter = 0
        calculated = False
        for geno in gp_map.map[pheno]:
            if len(connections[geno]) > 1:
                calculated = True
                counter += 1
                matching_phenotype = 0
                num_neighbours = len(connections[geno])
                for connected_to in connections[geno]:
                    if connected_to.phenotype.phenotype == geno.phenotype.phenotype:
                        matching_phenotype += 1
                
                pheno_robustness += (matching_phenotype/num_neighbours)
                
        if calculated:
            total += (pheno_robustness / counter)
            num_calculated += 1
            
    return total / num_calculated
    
def innov_rate(gp_map):
    """ 
    Calculates the innovation rate of a genotype-phenotype mapping,
    measuring the evolvability of a population in a given mapping.
    
    Taken from ebner et al. "How Neutral Networks Influence Evolvability"
    """
    total_encountered = []
    for phenotype in gp_map:
        encountered_per_step = [] * 100
        for _ in range(100): # 100 random walks for each phenotype
            random_genotype = choice(gp_map[phenotype])
            # Random neutral walk starting from genotype of size 100
            # Average result over all walks
            # Plot length of random neutral walk vs avg num phenotypes encountered
            # np.cumsum

def genotype_evolvability(gp_map) -> float:
    """
    taken from https://royalsocietypublishing.org/doi/pdf/10.1098/rspb.2007.1137. 
    Number of different phenotypes in the 1 mutation neighbourhood of a genotype
    """
    total = 0
    counter = 0
    connections = defaultdict(list)
    
    for connection in gp_map.connections:
        connections[connection.n1].append(connection.n2)
        connections[connection.n2].append(connection.n1)
    
    for pheno in gp_map.map:
        for geno in gp_map.map[pheno]:
            if len(connections[geno]) > 1:
                counter += 1
                for connected_to in connections[geno]:
                    if connected_to.phenotype.phenotype != geno.phenotype.phenotype:
                        total += 1
    
    return total / counter

def phenotype_evolvability(gp_map) -> float:
    """
    
    """
    total = 0
    counter = 0
    div = 0 
    connections = defaultdict(list)
    
    for connection in gp_map.connections:
        connections[connection.n1].append(connection.n2)
        connections[connection.n2].append(connection.n1)
    
    for pheno in gp_map.map:
        calculated = False
        for geno in gp_map.map[pheno]:
            if len(connections[geno]) > 1:
                calculated = True
                counter += 1
                for connected_to in connections[geno]:
                    if connected_to.phenotype.phenotype != geno.phenotype.phenotype:
                        total += 1
        
        if calculated:
            div += 1
    
    return total / div

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
        file.close()
        return obj


if __name__ == "__main__":   
    gp = load("genotype-phenotype_maps/CPPN-NEAT-GP-MAP-1-MIL.pickle")
    #gp = load("genotype-phenotype_maps/ES-HYPERNEAT-GP-MAP-1000000")
    
    print("LOADED")
    
    most_complex = gp.most_complex()
    
    print(most_complex[0].complexity)
    show(most_complex[0].phenotype)
    
    print(most_complex[-1].complexity)
    show(most_complex[-1].phenotype)
    
    print(most_complex[round(len(most_complex)/2)].complexity)
    show(most_complex[round(len(most_complex)/2)].phenotype)

#%%