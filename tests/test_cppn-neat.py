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
from networks import CPPN, NodeType
from tools.draw_cppn import draw_cppn

#TODO Add tests to test the functionality of cppn-neat
#TODO Add comments

def test_generate_population() -> None:
    """
    Tests the "create_population" function in cppn_neat.
    """
    population = cppn_neat.generate_population([8,8,7], 10)
    assert len(population) == 10

    for cppn in population:
        assert len(cppn.nodes[-1]) == 2 # Checks there are 2 outputs
        assert len(cppn.nodes[0]) == 5 # Checks there are 5 inputs
        assert cppn.material is None
        assert cppn.presence is None

        for connection in cppn.connections:
            assert connection.weight >= -1 and connection.weight <= 1

def test_layers_correct():
    population = cppn_neat.generate_population([8,8,7], 100)

    for cppn in population:
        for connection in cppn.connections:
            assert connection.out.layer < connection.input.layer

def test_crossover_indv() -> None:
    """
    
    """
    cppn_one = CPPN([8,8,7])
    cppn_two = CPPN([8,8,7])

    cppn_one.fitness = 15
    cppn_two.fitness = 10

    child = cppn_neat.crossover(cppn_one, cppn_two)
    assert not child.has_cycles()

    for connection in child.connections:
        assert connection.out.layer < connection.input.layer

def test_add_con_when_full() -> None:
    """
    
    """
    cppn_one = CPPN([8,8,7])

    before = len(cppn_one.connections)
    before_connections = cppn_one.connections

    cppn_neat.add_connection(cppn_one)

    assert len(cppn_one.connections) == before
    assert cppn_one.connections == before_connections

def test_add_con_when_valid():
    cppn = CPPN([8,8,7])

    cppn_neat.add_node(cppn)
    cppn_neat.add_node(cppn)
    cppn_neat.add_node(cppn)

    before = len(cppn.connections)

    cppn_neat.add_connection(cppn)

    num_nodes = 0
    for layer in cppn.nodes:
        num_nodes += len(layer)

    assert(len(cppn.nodes)) > 2
    assert num_nodes == 10
    assert len(cppn.connections) == before + 1

    for connection in cppn.connections:
        assert connection.out.layer < connection.input.layer

def test_add_multiple_con_when_valid():
    cppn = CPPN([8,8,7])

    for _ in range(10):
        cppn_neat.add_node(cppn)
    
    cons_before = len(cppn.connections)
    
    # for _ in range(10):
    #     cppn_neat.add_connection(cppn)
    
    # assert len(cppn.connections) == cons_before + 10

    for connection in cppn.connections:
        assert connection.out.layer < connection.input.layer

def test_add_node() -> None:
    """
    
    """
    cppn_one = CPPN([8,8,7])
    cppn_two = CPPN([8,8,7])

    previous_num_connections = len(cppn_one.connections)
    
    cppn_neat.add_node(cppn_one)
    total_num_nodes = 0

    for layer in cppn_one.nodes:
        total_num_nodes += len(layer)

    assert len(cppn_one.nodes) == 3
    assert total_num_nodes == 8
    assert len(cppn_one.nodes[0]) == 5
    assert len(cppn_one.nodes[1]) == 1
    assert len(cppn_one.nodes[2]) == 2

    for node in cppn_one.nodes[0]:
        assert node.type in [NodeType.INPUT_B, NodeType.INPUT_D, NodeType.INPUT_X, NodeType.INPUT_Y, NodeType.INPUT_Z]
    
    for node in cppn_one.nodes[1]:
        assert node.type is NodeType.HIDDEN
    
    for node in cppn_one.nodes[1]:
        assert node.layer == 1
    
    for node in cppn_one.nodes[2]:
        assert node.type in [NodeType.MATERIAL_OUTPUT, NodeType.PRESENCE_OUTPUT]
    
    num_enabled = 0

    for connection in cppn_one.connections:
        if connection.enabled:
            num_enabled += 1
    
    assert num_enabled == previous_num_connections + 1

    innov_numbers_one = set()
    innov_numbers_two = set()
    
    for connection in cppn_one.connections:
        innov_numbers_one.add(connection.historical_marking)
    
    for connection in cppn_two.connections:
        innov_numbers_two.add(connection.historical_marking)
    
    assert innov_numbers_one != innov_numbers_two

def test_add_multiple_nodes():
    cppn = CPPN([8,8,7])

    for _ in range(100):
        cppn_neat.add_node(cppn)

    num_nodes = 0
    for layer in cppn.nodes:
        num_nodes += len(layer)
    
    draw_cppn(cppn)
    
    assert num_nodes == 107
    assert len(cppn.nodes) > 2

    counter = 0
    for layer in cppn.nodes:
        for node in layer:
            assert node.layer == counter
        
        counter += 1
    
    for connection in cppn.connections:
        assert connection.out.layer < connection.input.layer
