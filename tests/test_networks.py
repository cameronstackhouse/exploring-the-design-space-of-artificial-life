"""
Module to test the functionality of the classes and methods in networks.py.
Run using py.test.
"""

import os, sys

#Adds parent directory to system path
p = os.path.abspath('.')
sys.path.insert(1, p)

import networks
from tools.activation_functions import sigmoid

#TODO Add tests to test the functionality of networks.py
def test_basic_cppn() -> None:
    """
    Method to test that a CPPN is initilised correctly with the
    correct number of nodes, layers, connections, and valid weights
    """
    cppn = networks.CPPN([1,1,1])
    assert len(cppn.nodes) == 2 #Checks that there are 2 layers in the CPPN
    assert len(cppn.nodes[0]) == 5 #Checks that there are 5 input nodes
    assert len(cppn.nodes[1]) == 2 #Checks that there are 2 output nodes
    assert len(cppn.connections) == 10 #Checks that there are 10 connections

    for connection in cppn.connections:
        assert (connection.weight >= 0 and connection.weight <= 1) #Checks that the connection weights are valid
    
    for node in cppn.nodes[1]:
        assert node.activation_function is sigmoid #Checks that the output nodes have sigmoid functions as there activation functions

def test_create_node() -> None:
    """
    Function to test creating a new node and adding it to
    a CPPN
    """
    cppn = networks.CPPN([1,1,1]) #Creates a basic CPPN
    networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 0) #Adds node to the first layer in the CPPN

    assert len(cppn.nodes) == 2
    assert len(cppn.nodes[0]) == 6

def test_create_connection() -> None:
    """
    
    """
    cppn = networks.CPPN([1,1,1])
    a = networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 0) #Adds node to the first layer in the CPPN
    b = networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 1) #Adds node to the second layer in the CPPN
    cppn.create_connection(a, b, 0.5)
    assert len(cppn.connections) == 11

def test_activate_basic_cppn() -> None:
    """

    """
    #TODO
    cppn = networks.CPPN([1,1,1])
    a = cppn.to_phenotype()
    assert a is not None

def test_activate_after_unconnected_node():
    """
    
    """
    #TODO
    cppn = networks.CPPN([1,1,1])
    a = cppn.to_phenotype()

    networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 0)

    b = cppn.to_phenotype()
    assert a == b

def test_activate_after_nodes_and_connection():
    """
    
    """
    #TODO
    cppn = networks.CPPN([1,1,1])
    a = cppn.to_phenotype()

    a = networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 0)
    cppn.create_connection(cppn.nodes[0][1], a, 0.3)
    cppn.create_connection(a, cppn.nodes[1][0], 0.2)

    b = cppn.to_phenotype()

    assert a != b

def test_reset() -> None:
    """
    
    """
    #TODO
    cppn = networks.CPPN([1,1,1])
    cppn.to_phenotype()
    cppn.reset()
    assert cppn.presence is None
    assert cppn.material is None

    for layer in cppn.nodes:
        for node in layer:
            assert node.output is None

def test_to_phenotype() -> None:
    #TODO
    pass


