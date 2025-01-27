""" 
Module implementing the ability to generate and analyse genotype-phenotype maps
for the production of xenobots
"""
# %%``
import os
import pickle
import neat
import json
import csv
from math import log10
from copy import deepcopy
from random import choice
from typing import Tuple, List, Dict, Set
from collections import defaultdict
import numpy as np
from tools.activation_functions import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs
from tools.gen_phenotype_from_genotype import genotype_to_phenotype
from tools.phenotype_information import calc_KC, movement_components
from tools.read_files import read_sim_output
from visualise_xenobot import show
from voxcraftpython.VoxcraftVXD import VXD
from voxcraftpython.VoxcraftVXA import VXA
from pureples.shared.substrate import Substrate
from pureples.es_hyperneat.es_hyperneat import ESNetwork
import seaborn as sns
import matplotlib.pyplot as plt

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
        self.num_cppn_designers = None
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
        Initilises a genotype-phenotype map object
        
        :param config_name: name of the config file for CPPN-NEAT or ES-HyperNEAT
        :param hyperneat: boolean indicating if the GP map is for ES-HyperNEAT
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
        Initilises the GP map given a number of genotypes
        
        :param num_genotypes: number of genotypes
        """
        self.gen_genotypes(num_genotypes)
        self.gen_phenotype_info()
        
    def gen_genotypes(
        self, 
        num_genotypes: int
        ) -> None:
        """
        Function to generate genotypes given a number of genotypes 
        
        :param num_genotypes: size of genotypes to generate (number of nodes)
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
        
        :rtype: Tuple(List, List)
        :return: 2 lists, list of genotypes and list of phenotypes
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
    
    def gen_phenotype_info(self) -> None:
        """ 
        Function to generate information about phenotypes
        within the genotype_phenotype map 
        """
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
        
        :rtype: List
        :return: Most designable phenotypes in decending order
        """
        sorted_phenotypes = sorted(self.phenotypes, key=lambda x: x.genotypes, reverse=True)
        return sorted_phenotypes

    def most_complex(self) -> List:
        """
        Returns a sorted list of phenotypes by their complexity,
        as calculated via the LZW algorithm
        
        :rtype: List
        :return: Most complex phenotypes in decending order
        """
        sorted_phenotypes = sorted(self.phenotypes, key=lambda x: x.complexity, reverse=True)
        return sorted_phenotypes

    def mean_genotype_to_phenotype_ratio(self) -> float:
        """ 
        Calculate the mean ratio of genotypes to phenotypes
        
        :rtype: float
        :return: mean genotype to phenotype ratio
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

def gen_motifs(phenotypes: List) -> Set:
    """ 
    Generates all 3x3x3 structural 
    motifs present in xenobots in the GP map.
            
    :rtype: set
    :return: set of motifs identified in the population of xenobots
    """
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
    phenotypes: List, 
    motifs: Set,
    motif_index: Dict
    )-> None:
    """ 
    Counts the number of 3x3x3 motifs in each phenotype 
    given a list of motifs.
            
    :param phenotypes: list of 
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

def gen_motif_clustering_data_file(phenotypes: List, motif_name: str) -> None:
    """ 
    Generate a CSV file containing motif information associated to each phenotype
    
    :param phenotypes: list of phenotypes 
    :param motif_name: name of file containing structural motif count data
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

def gen_behaviour_clustering_data_file(behaviour_comps: List, name: str) -> None:
    """ 
    Function to generate a CSV file given a list of behaviour components for each xenbot
    
    :param behaviour_comps: behavioural components of xenobots
    :param name: name of csv file
    """
    headers = ["X1", "X2", "X3", "X4", "Y1", "Y2", "Y3", "Y4", "Z1", "Z2", "Z3", "Z4"]
    with open(name, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(headers)
        
        for row in behaviour_comps:
            writer.writerow(row)
        
def complexity_probability_distribution(gp_map: GenotypePhenotypeMap) -> None:
    """ 
    Function to plot the complexity-probability distribution of a population 
    of xenobots
    
    :param gp_map: Genotype-Phenotype map
    """
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

def robustness(gp_map: GenotypePhenotypeMap) -> float:
    """ 
    Calculates the mean genotypic robustness of a genotype-phenotype mapping,
    meaning the mean number of genotype mutations which do not 
    alter the resulting phenotype structure.
    
    :param gp_map: genotype-phenotype map
    :rtype: float
    :return: genotypic robustness
    """
    
    connections = defaultdict(list)
    
    for connection in gp_map.connections:
        connections[connection.n1].append(connection.n2)
        connections[connection.n2].append(connection.n1)
    
    counter = 0
    total = 0
    for phenotype in gp_map.map:
        # Inner loop to calculate the ratio of genotype mutations which leave the phenotype unchanged
        for genotype in gp_map.map[phenotype]:
            if len(connections[genotype]) > 1:
                counter += 1
                matching_phenotype = 0
                num_neighbours = len(connections[genotype])
                for connected_to in connections[genotype]:
                    if connected_to.phenotype.phenotype == genotype.phenotype.phenotype:
                        matching_phenotype += 1
                
                total += (matching_phenotype/num_neighbours)
    
    # Mean of the total ratio
    return total / counter

def phenotypic_robustness(gp_map: GenotypePhenotypeMap) -> float:
    """ 
    Average of genotypic robustness over genotypes that match to that phenotype
    
    :param gp_map: genotype-phenotype map
    :rtype: float
    :return: phenotypic robustness
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
    
def genotype_evolvability(gp_map: GenotypePhenotypeMap) -> float:
    """
    taken from https://royalsocietypublishing.org/doi/pdf/10.1098/rspb.2007.1137. 
    Number of different phenotypes in the 1 mutation neighbourhood of a genotype
    
    :param gp_map: genotype-phenotype map
    :rtype: float
    :return: genotypic evolvability 
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

def phenotype_evolvability(gp_map: GenotypePhenotypeMap) -> float:
    """
    Calculates the phenotypic evolvability, defined as the average 
    number of phenotypes reachable from every phenotype
    
    :param gp_map: genotype-phenotype map
    :rtype: float
    :return: mean phenotypic evolvability
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

def calculate_frequency_components(phenotypes: List) -> List:
    """ 
    Function to calculate the frequency components of the 
    movement of a list of xenobots.
    
    :param phenotypes: List of xenobot phenotypes
    :rtype: List
    :return: List of frequency components for each phenotype
    """
    vectors = []
    for phenotype in phenotypes:
        body = np.array(phenotype.phenotype)
        shaped = body.reshape(8,8,7)
        frequency_comp = movement_components(shaped)

        x = list(frequency_comp[0][:4])
        y = list(frequency_comp[1][:4])
        z = list(frequency_comp[2][:4])
        
        components = np.append(x, np.append(y, z))

        vectors.append(components)
        print(len(vectors))
        
    return vectors

def calculate_fitness(phenotypes: List) -> None:
    """ 
    Calculates and assigns fitness to each xenobot based on the 
    centre of mass displacement over the course of three seconds.
    
    :param phenotypes: list of xenobot bodies to evaluate.
    """
        
    fitness_vals = np.zeros(len(phenotypes))
                
    vxa = VXA(SimTime=3, HeapSize=0.65, RecordStepSize=100, DtFrac=0.95, EnableExpansion=1)
    passive = vxa.add_material(RGBA=(0,255,0), E=5000000, RHO=1000000) # passive soft
    active = vxa.add_material(RGBA=(255,0,0), CTE=0.01, E=5000000, RHO=1000000) # active
        
    os.system(f"rm -rf gp-fitness/") #Deletes contents of run directory if exists
    os.system(f"mkdir -p gp-fitness") # Creates a new directory to store fitness files
    vxa.write("base.vxa") #Write a base vxa file
    os.system(f"cp base.vxa gp-fitness") #Copy vxa file to correct run directory
    os.system("rm base.vxa") #Removes old vxa file
                
    evaluated = 0
            
    for phen_index, phenotype in enumerate(phenotypes):
        evaluated += 1
        #fitness_file_mapping[phen_index] = phenotype
            
        vxd = VXD()
        vxd.set_tags(RecordVoxel=1)
        body = np.zeros(8*8*7)
            
        for cell in range(len(phenotype.phenotype)):
            if phenotype.phenotype[cell] == 1:
                body[cell] = passive
            elif phenotype.phenotype[cell] == 2:
                body[cell] = active 
            
        reshaped = body.reshape(8,8,7)
        vxd.set_data(reshaped)
            
        vxd.write(f"id{phen_index}.vxd") #Writes vxd file for individual
        os.system(f"cp id{phen_index}.vxd gp-fitness/")
        os.system(f"rm id{phen_index}.vxd") #Removes the old non-copied vxd file
            
        if (evaluated >= 100) or (phen_index == len(phenotypes) - 1):
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
            
                #Assign absolute movement fitness value to phenotype
                fitness_vals[phenotype_index] = float(result["fitness"])
                    
                    # for genotype in phenotypes[fitness_file_mapping[phenotype_index]]:
                    #     genotype.phenotype.fitness["abs_movement"] = float(result["fitness"])
                
            evaluated = 0
    
    os.system("rm -rf gp-fitness")
    
    return fitness_vals


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
    #gp = load("genotype-phenotype_maps/CPPN-NEAT-GP-MAP-1-MIL.pickle")
    #gp = load("genotype-phenotype_maps/ES-HYPERNEAT-GP-MAP-1000000")
    
    # Creation of GP map
    demo_map = GenotypePhenotypeMap("config-gpmap")
    demo_map.initilise(1000)
    
    hyperneat_demo_map = GenotypePhenotypeMap("config-hyperneat-gpmap", hyperneat=True)
    
    # Robustness and evolvability
    phenotypic_robustness = phenotypic_robustness(demo_map)
    genotypic_robustness = robustness(demo_map)
    genotypic_evolvability = genotype_evolvability(demo_map)
    phenotypic_evolvability = phenotype_evolvability(demo_map)
    
    
    print(f"Phenotypic robustness: {phenotypic_robustness}\nGenotypic robustness: {genotypic_robustness}\nPhenotypic Evolvability: {phenotypic_evolvability}\nGenotypic Evolvability: {genotypic_evolvability}")
    
    
    motifs = gen_motifs(demo_map.phenotypes)
    frequency_components = calculate_frequency_components(demo_map.phenotypes)
    
    print(motifs)
    complexity_probability_distribution(demo_map)
#%%