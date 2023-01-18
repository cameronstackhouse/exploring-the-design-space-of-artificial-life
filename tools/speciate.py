from typing import List

"""
Module containing all functions relating to the speciation of 
Xenobots using functions defined by Stanley in the 
"Evolving Neural Networks through Augmenting Topologies" paper.
"""

def speciate(
    pop: List, 
    threshold: float
    ) -> None:
    """
    Function to speciate a population, updating fitness scores of xenobots accordingly
    based on which species they belong in, protecting early topological innovation.

    :param pop: Population of xenobots
    """

    # Updated fitness = fitness / sum of sharing function to each other individual in population
    for individual in pop:
        # TODO Add comments
        individual.fitness = individual.fitness / sum([share(cppn_distance(individual, compare), threshold) for compare in pop])

def cppn_distance(
    cppn1, 
    cppn2
    ) -> float:
    """
    Function to find the "distance" between two cppns to determine
    which species a geneotype belongs in for the CPPN NEAT algorithm

    :param cppn1: first cppn to compare
    :param cppn2: second cppn to compate
    :rtype: float
    :return: distance between the two cppns
    """

    #TODO Verify this! (pg 13: https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

    disjoint_counter = 0 #Counter of disjoint connections
    excess_counter = 0 #Counter of excess connections
    weight_value_one = 0 #Summation value of matching weights in the first cppn
    weight_value_two = 0 #Summation value of matching weights in the second cppn

    max_cppn2_innov = 0 #Max innovation number of connections in cppn2

    #Finds the maximum innovation number connection in the second cppn
    for connection in cppn2.connections:
        if connection.historical_marking > max_cppn2_innov:
            max_cppn2_innov = connection.historical_marking
    
    #Iterates through connections in the first cppn, incrementing the excess counter if the innovation number is higher than the max historical marking in the second cppn
    for connection in cppn1.connections:
        if connection.historical_marking > max_cppn2_innov:
            excess_counter+=1
        
    #Iterates through connections in both cppns, summing the value of weights of matching connections and incrementing the disjoint counter if they don't match
    for connection1 in cppn1.connections:
        shared = False
        for connection2 in cppn2.connections:
            if connection1.historical_marking == connection2.historical_marking:
                shared = True
                weight_value_one += connection1.weight
                weight_value_two += connection2.weight
                break
        
        if not shared:
            disjoint_counter+=1
    
    print(disjoint_counter)
    print(f"weight vals: {weight_value_one} {weight_value_two}")
    #TODO Return function val

def share(
    cppn_distance: float, 
    threshold: float
    ) -> int:
    """
    Share function defined by Stanley to measure if two networks are in the 
    same species for the NEAT algorithm.

    :param cppn_distance: distance between the two CPPNs
    :param threshold: threshold determining if two networks are in the same species
    :return: integer indicating if the two networks belong in the same species
    """
    if cppn_distance > threshold: #Checks if CPPN distance is greater than the threshold
        return 0 #If so the CPPNs are not in the same species
    else:
        return 1 #Otherwise the CPPNs are in the same species

def distance(
    cppn1, 
    cppn2, 
    c1: float, 
    c2: float, 
    c3: float
    ) -> float:
    """
    Function to determine the distance between two CPPNs, measuring their similarity.

    :param cppn1: first CPPN to compare
    :param cppn2: second CPPN to compare
    :param c1: weight of excess genes
    :param c2: weight of disjoint genes
    :param c3: weight of average weight differences of matching genes

    :return: distance between the two CPPNs using the function defined by Stanley in the NEAT paper
    """
    # E: Number of excess genes
    # D: Number of disjoint genes
    # W: Average weight differences of matching genes (Including disabled genes!)
    # distance = c1*E / N + c2*D / N + c3 * W

    N = max(len(cppn1.connections), len(cppn2.connections)) # Gets the number of genes in the larger genome
    excess = 0 # Initilises excess genes counter
    disjoint = 0 # Initilises disjoint genes counter
    avg_weight = 0 

    # Gets the number of excess genes

    # Gets the number of disjoint genes

    # Gets the average weight of matching connections

    return (c1*excess/N) + (c2*disjoint/N) + c3*avg_weight