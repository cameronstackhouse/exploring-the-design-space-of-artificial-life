from typing import List

"""
Module containing all functions relating to the speciation of 
Xenobots using functions defined by Stanley in the 
"Evolving Neural Networks through Augmenting Topologies" paper.
"""

#TODO Compare with other implementations
#TODO Dynamic threshold maybe for limiting number of species

def speciate(
    pop: List, 
    threshold: float
    ):
    """
    Function to speciate a population, updating fitness scores of xenobots accordingly
    based on which species they belong in, protecting early topological innovation.

    :param pop: Population of xenobots
    """
    #TODO Verify this
    for indv in pop:
        denom = sum([share(distance(indv, compare), threshold) for compare in pop])
        indv.fitness = indv.fitness / denom
    
    split = split_into_species(pop, threshold)
    
    return split

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

    # TODO Finish

    excess = 0
    disjoint = 0
    avg_matching_weights = 0
    matching_genes = 0
    N = max(len(cppn1.connections), len(cppn2.connections))

    if N < 20:
        N = 1
    
    for connection in cppn1.connections:
        for connection_b in cppn2.connections:
            if connection.historical_marking == connection_b.historical_marking:
                avg_matching_weights += abs(connection.weight) - abs(connection_b.weight)
                matching_genes += 1
    
    markings_one = set()
    markings_two = set()

    for connection in cppn1.connections:
        markings_one.add(connection.historical_marking)
    
    for connection in cppn2.connections:
        markings_two.add(connection.historical_marking)
    
    smallest = min(max(markings_one), max(markings_two))

    one_two_dif = markings_one - markings_two
    two_one_dif = markings_two - markings_one

    for marking in one_two_dif:
        if marking > smallest:
            excess += 1
        else:
            disjoint += 1
    
    for marking in two_one_dif:
        if marking > smallest:
            excess += 1
        else:
            disjoint += 1
    
    avg_matching_weights = avg_matching_weights / matching_genes

    return (c1*excess / N) + (c2*disjoint / N) + (c3 * avg_matching_weights)

def split_into_species(population, threshold):
    """
    Splits a population into species.
    This is used for selection of individuals for crossover,
    allowing for topological innovation by protecting less fit
    individuals in small populations.

    :param population: population of CPPNs to split into species
    """
    species = [[population[0]]]

    for indivdual in population[1:]:
        placed = False

        for spec in species:
            for representative in spec:
                if share(distance(indivdual, representative), threshold) == 1:
                    placed = True
                    spec.append(indivdual)
                    break
        
            if placed:
                break
        
        if not placed:
            species.append([indivdual])
    
    return species

