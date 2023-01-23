"""
Module to test the functionality of the classes and methods in networks.py.
Run using pytest.
"""

import os, sys

#Adds parent directory to system path
p = os.path.abspath('.')
sys.path.insert(1, p)

import numpy as np
import networks
from tools.activation_functions import sigmoid, neg_abs, neg_square, sqrt_abs, neg_sqrt_abs, normalize

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
        assert (connection.weight >= -1 and connection.weight <= 1) #Checks that the connection weights are valid
    
    for node in cppn.nodes[1]:
        assert node.activation_function is sigmoid #Checks that the output nodes have sigmoid functions as there activation functions

def test_set_activation_function() -> None:
    """
    Tests changing the activation function of a node
    """
    cppn = networks.CPPN([1,1,1]) # Creates a CPPN for the node to be added to
    node = networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 0) # Creates a new node with a sigmoid activation function
    node.set_activation_function(neg_abs) # Changes the activation function to negative absolute
    assert node.activation_function is not sigmoid and node.activation_function is neg_abs # Checks activation function has been changed

def test_activate() -> None:
    """
    Tests activating a singular node to see if an output is produced
    """
    #TODO
    cppn = networks.CPPN([1,1,1])
    node = networks

def test_add_node() -> None:
    """
    Function to test creating a new node and adding it to
    a CPPN
    """
    cppn = networks.CPPN([1,1,1]) #Creates a basic CPPN
    node = networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 0) #Adds node to the first layer in the CPPN

    assert len(cppn.nodes) == 2 #Asserts that there are only two layers
    assert len(cppn.nodes[0]) == 6 #Asserts that the first layer now has six nodes
    assert node.position == 5 #Asserts that the position of the node has been properly set

    # Asserts the positions of the upper level nodes have been updated
    assert cppn.nodes[-1][0].position == 6
    assert cppn.nodes[-1][1].position == 7

def test_remove_node() -> None:
    """
    Tests the removal of a node, ensuring that the node has successfully been removed 
    and that positions of other nodes have not been changed
    """
    cppn = networks.CPPN([8,8,7])
    node = networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 1)

    before = cppn.to_phenotype()

    assert node.position == 7

    cppn.remove_node(node)

    assert len(cppn.nodes[1]) == 2

    # Ensures the positions have been updated of surrounding nodes accordingly 
    for i in range(5):
        assert cppn.nodes[0][i].position == i
    
    for i in range(2):
        assert cppn.nodes[1][i].position == i + 5
    
    after = cppn.to_phenotype()

    assert np.array_equal(before, after)

def test_remove_node_invalid() -> None:
    """
    Tests the removal of a node 
    """
    cppn = networks.CPPN([8,8,7])

    cppn.remove_node(cppn.nodes[0][0])

    assert len(cppn.nodes[0]) == 5

def test_create_connection() -> None:
    """
    Function to test creating a connection in the CPPN
    """
    cppn = networks.CPPN([1,1,1])
    cppn_2 = networks.CPPN([1,1,1])

    a = networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 0) #Adds node to the first layer in the CPPN
    b = networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 1) #Adds node to the second layer in the CPPN

    a_2 = networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn_2, 0)
    b_2 = networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn_2, 1)

    cppn.create_connection(a, b, 0.5) # Adds a connection between the two nodes with a weight of 0.5
    assert len(cppn.connections) == 11 # Asserts that a connection has been successfully created

    #TODO Assert that it is in the collection of connections
    #TODO Add the same connection in CPPN and assert that it is not added to list of conenctions as already exists
    #TODO Also check to make sure the innovation numbers are the same

def test_set_input_states() -> None:
    """
    Tests the set_input_states function to set the correct
    range of inputs to the CPPN
    """
    cppn = networks.CPPN([1,1,1])

    # Asserts that the CPPN has the correct inputs
    assert len(cppn.x_inputs) > 0
    assert len(cppn.y_inputs) > 0
    assert len(cppn.z_inputs) > 0
    assert len(cppn.d_inputs) > 0
    assert len(cppn.b_inputs) > 0

def test_activate_basic_cppn() -> None:
    """
    Tests activating a basic initilized CPPN
    """
    cppn = networks.CPPN([1,1,1]) #Creates the CPPN
    a = cppn.to_phenotype()
    assert a is not None #Asserts that a phenotype is produced

def test_activate_after_unconnected_node() -> None:
    """
    Tests activating the CPPN after adding a non-connected node
    """
    cppn = networks.CPPN([1,1,1])
    a = cppn.to_phenotype()

    networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 0) #Adds new node which is unconnected

    b = cppn.to_phenotype()
    assert a == b #Asserts that the new phenotype is the same as the old, as the node added is not connected to any others

def test_activate_after_nodes_and_connection() -> None:
    """
    Function to test the activation of nodes after adding an extra node 
    and adding extra connections. The phenotype produced should be different
    to the one produced before the addition of the node and connections
    """
    cppn = networks.CPPN([1,1,1]) #Creates a basic CPPN
    a = cppn.to_phenotype() #Converts the CPPN to phenotype

    a = networks.Node(sigmoid, networks.NodeType.HIDDEN, cppn, 0) #Adds a new node to the CPPN at layer 0
    #Creates new connections between nodes
    cppn.create_connection(cppn.nodes[0][1], a, 0.3)
    cppn.create_connection(a, cppn.nodes[1][0], 0.2)

    b = cppn.to_phenotype()

    assert a != b

def test_reset() -> None:
    """
    Tests the reset function for the CPPN, ensuring
    that each nodes output has been reset back to having no value
    """
    cppn = networks.CPPN([1,1,1]) # Creates a CPPN
    cppn.to_phenotype() # Activates the CPPN by 
    cppn.reset() # Resets the CPPN

    # Asserts that the CPPN output is none
    assert cppn.presence is None
    assert cppn.material is None

    # Asserts that the node output for each node is none
    for layer in cppn.nodes:
        for node in layer:
            assert node.output is None

def test_to_phenotype() -> None:
    """
    Tests the to_phenotype() function which activates
    a CPPN to produce the xenobot phenotype, ensuring that 
    the xenobot produced is of the correct size and has the correct
    type of cells
    """

    # Creates two CPPNs with different design space sizes
    cppn = networks.CPPN([8,8,7]) 
    cppn2 = networks.CPPN([2,2,2])

    # Asserts that the phenotype of each is the correct size 
    assert len(cppn.to_phenotype()) == 448
    assert len(cppn2.to_phenotype()) == 8

    # Asserts that the phenotypes produced have only contain valid cells
    valid = True
    for cell in cppn.to_phenotype():
        if cell not in [0,1,2]:
            valid = False
            break
    
    assert valid is True

def test_run() -> None:
    """
    Tests the running of a compositional pattern producing network to ensure
    material and presence output values are produced.
    """
    cppn = networks.CPPN([8,8,7])
    cppn.run(0) # Runs the CPPN taking the 0th pixels coordinates as input
    
    # Asserts that an output is produced in both material and presence of 
    # a cell in the given pixel location (0)
    assert cppn.material is not None 
    assert cppn.presence is not None
