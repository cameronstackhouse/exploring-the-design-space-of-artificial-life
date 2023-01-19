"""
Module to test the functionality of cppn-neat.py
using pytest.
"""
import os, sys
from random import randint

#Adds parent directory to system path
p = os.path.abspath('.')
sys.path.insert(1, p)

import cppn_neat
from networks import CPPN

#TODO Add tests to test the functionality of cppn-neat
#TODO Add comments

def test_evolve() -> None:
    """
    
    """
    pass

def test_initial_mutations() -> None:
    """
    
    """
    pass

def test_create_population() -> None:
    """
    Tests the "create_population" function in cppn_neat.
    """
    population = cppn_neat.create_population(10, [8,8,7])
    assert len(population) == 10

    for cppn in population:
        assert len(cppn.nodes[-1]) == 2 # Checks there are 2 outputs
        assert len(cppn.nodes[0]) == 5 # Checks there are 5 inputs
        assert cppn.material is None
        assert cppn.presence is None

        for connection in cppn.connections:
            assert connection.weight >= -1 and connection.weight <= 1

def test_select_population() -> None:
    """
    Tests the "select_population" function in cppn_neat.
    """
    # Creates population of CPPNs
    population = cppn_neat.create_population(10, [8,8,7])
    
    # Assigns a psuedo fitness to each CPPN
    for cppn in population:
        cppn.fitness = randint(1, 10)

    # Gets the top 5 and bottom 5 CPPNs 
    sorted_pop = sorted(population, key=lambda indv: indv.fitness)
    top_5 = sorted_pop[:5]
    bottom_5 = sorted_pop[5:]
    
    # Selects the top 50% of CPPNs according to fitness
    population = cppn_neat.select_population(population, 10, 0.5)

    assert len(population) == 5 # Asserts that only 5 CPPNs have been selected
    
    # Asserts that the top 5 fittest CPPNs are still in the population
    for cppn in top_5:
        assert cppn in population
    
    # Asserts that the bottom 5 least fit CPPNs are no longer in the population
    for cppn in bottom_5:
        assert cppn not in population

def test_crossover_indv() -> None:
    """
    
    """
    cppn_one = CPPN([8,8,7])

    cppn_one.fitness = 15

    print(CPPN.innovation_counter)

def test_crossover_pop() -> None:
    """
    
    """
    pass

def test_mutate_node() -> None:
    """
    
    """
    pass

def test_mutate_nodes() -> None:
    """
    
    """
    pass

def test_add_node_between_con() -> None:
    """
    
    """
    pass

def test_add_node_pop() -> None:
    """
    
    """
    pass

def test_mutate_connection() -> None:
    """
    
    """
    pass

def test_mutate_connections() -> None:
    """
    
    """
    pass

def test_remove_connection() -> None:
    """
    
    """
    pass

def test_remove_connections() -> None:
    """
    
    """
    pass

def test_add_connection() -> None:
    """
    
    """
    pass

def test_add_connections() -> None:
    """
    
    """
    pass

def test_remove_nodes() -> None:
    """
    
    """
    pass

def test_mutate_population() -> None:
    """
    
    """
    pass

test_crossover_indv()