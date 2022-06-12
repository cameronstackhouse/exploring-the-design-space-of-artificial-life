#CHECK WHAT ACTIVATION FUNCTIONS USED IN PAPER

from random import choice
from typing import Callable
from enum import Enum
from utilities.activation_functions import sigmoid, periodic, identity, gaussian, repeat_asym, absolute, inverse, symmetric #Imports all activation functions

"""
Module defining components for the creation of functioning
compositional pattern-producing networks
"""

class NodeType(Enum):
    """
    Class defining the three different node types in a CPPN - Input, Hidden, and Output
    """
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class Node:
    """
    Class defining a node in a compositional pattern-producing network
    """
    def __init__(self, activation_function: Callable, type: NodeType) -> None:
        self.inputs = [] #Input values passed into the node
        self.activation_function = activation_function #Activation function of the node
        self.type = type #Type of node (Input, Hidden, Output)
        self.output = 0 #Initilises the node output to 0
    
    def set_activation_function(self, activation_function: Callable) -> None:
        """
        Function to set the activation function of a node

        """
        self.activation_function = activation_function
    
    def add_input(self, value: float) -> None:
        """
        Function to add a value to the input values into a node

        """
        #ENSURE THAT VALUE PASSED IN HAS BEEN MULTIPLIED BY WEIGHT OF CONNECTION
        #TO BE DELETED!!!!
        self.inputs.append(value)
    
    def activate(self) -> None:
        """
        Function to sum the input values into the node and 
        pass the total into the nodes activation function
        """
        total = 0
        for value in self.inputs:
            total += value #Sums the input values
        
        self.output = self.activation_function(total) #Sets the output value to the activation function applied on the summation of input values


class CPPN:
    """
    Class defining a compositional pattern-producing network made of
    interconnected nodes with varying activation functions
    """
    def __init__(self) -> None:
        """
        
        """
        self.activation_functions = [sigmoid, periodic, identity, gaussian, repeat_asym, absolute, inverse, symmetric] #List of possible activation functions for each node in the network
        self.nodes = [] #List of nodes in the network
        self.connections = [] #List of connections between nodes in the network
        self.innovation_counter = 0 #Innovation counter for adding new connections to the network
    
    def reset(self) -> None:
        """
        Clears input and output values of each node in the network
        """
        for node in self.nodes:
            node.inputs = []
            node.output = 0
    
    
if __name__ == "__main__":
    """
    ******************
    DELETE FOR RELEASE
    TESTING ZONE
    ******************
    """