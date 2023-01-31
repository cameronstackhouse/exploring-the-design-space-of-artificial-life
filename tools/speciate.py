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
    #TODO CHECK THIS TOO

    # Updated fitness = fitness / sum of sharing function to each other individual in population
    for individual in pop:
        # TODO Add comments
        divider = sum([share(distance(individual, compare), threshold) for compare in pop])

        if divider != 0:
            individual.fitness = individual.fitness / divider

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

    #TODO Make sure this works
    #TODO Add comments
    #NOTE: Also include an additional argument which counts how many activation functions differ between 2 individuals

    N = max(len(cppn1.connections), len(cppn2.connections)) # Gets the number of genes in the larger genome
    excess = 0 # Initilises excess genes counter
    disjoint = 0 # Initilises disjoint genes counter
    weight_difference = 0 
    differing_activation_functions = 0

    innov_numbers_1 = set()
    innov_numbers_2 = set()
    for connection in cppn1.connections:
        innov_numbers_1.add(connection.historical_marking)
    
    for connection in cppn2.connections:
        innov_numbers_2.add(connection.historical_marking)

    # Gets the number of excess genes
    #TODO Add comments and verify
    max_one = max(innov_numbers_1)
    max_two = max(innov_numbers_2)

    if max_one > max_two:
        for gene in innov_numbers_1:
            if gene > max_two:
                excess += 1
    else:
        for gene in innov_numbers_2:
            if gene > max_one:
                excess += 1

    # Gets the number of disjoint genes
    one_minus_two = innov_numbers_1.difference(innov_numbers_2)
    two_minus_one = innov_numbers_2.difference(innov_numbers_1)

    if max_one > max_two:
        for gene in innov_numbers_2:
            if gene not in innov_numbers_1:
                disjoint += 1
        
        for gene in innov_numbers_1:
            if gene not in innov_numbers_2 and gene < max_two:
                disjoint += 1
    else:
        for gene in innov_numbers_1:
            if gene not in innov_numbers_2:
                disjoint += 1
        
        for gene in innov_numbers_2:
            if gene not in innov_numbers_1 and gene < max_one:
                disjoint += 1

    # Gets the weight difference of matching connections
    for connection in cppn1.connections:
        for connection_2 in cppn2.connections:
            if connection.historical_marking == connection_2.historical_marking:
                weight_difference += abs(connection.weight - connection_2.weight)
    
    # Gets the number of differing activation functions between the CPPNs
    cppn_one_activation_functions = {}
    cppn_two_activation_functions = {}

    for activation_function in cppn1.activation_functions:
        if activation_function in cppn_one_activation_functions:
            cppn_one_activation_functions[activation_function] += 1
        else:
            cppn_one_activation_functions[activation_function] = 1
    
    for activation_function in cppn2.activation_functions:
        if activation_function in cppn_two_activation_functions:
            cppn_two_activation_functions[activation_function] += 1
        else:
            cppn_two_activation_functions[activation_function] = 1 
    
    for activation_function in cppn_one_activation_functions:
        differing_activation_functions += abs(cppn_two_activation_functions[activation_function] - cppn_one_activation_functions[activation_function])

    return (c1*excess/N) + (c2*disjoint/N) + c3*weight_difference + differing_activation_functions