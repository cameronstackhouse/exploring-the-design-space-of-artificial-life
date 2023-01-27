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
        individual.fitness = individual.fitness / sum([share(distance(individual, compare), threshold) for compare in pop])

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
    c1: float = 1, 
    c2: float = 1, 
    c3: float = 1
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

    #TODO Program this
    #NOTE: Also include an additional argument which counts how many activation functions differ between 2 individuals

    N = max(len(cppn1.connections), len(cppn2.connections)) # Gets the number of genes in the larger genome
    excess = 0 # Initilises excess genes counter
    disjoint = 0 # Initilises disjoint genes counter
    avg_weight = 0 

    # Gets the number of excess genes

    # Gets the number of disjoint genes

    # Gets the average weight of matching connections

    return (c1*excess/N) + (c2*disjoint/N) + c3*avg_weight