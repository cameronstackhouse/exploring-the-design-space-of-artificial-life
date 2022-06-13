#CHECK WHAT ACTIVATION FUNCTIONS USED IN PAPER

from random import choice
from typing import Callable
from enum import Enum
from queue import PriorityQueue
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

class CPPN:
    """
    Class defining a compositional pattern-producing network made of
    interconnected nodes with varying activation functions
    """
    def __init__(self) -> None:
        """
        
        """
        self.activation_functions = [sigmoid, periodic, identity, gaussian, repeat_asym, absolute, inverse, symmetric] #List of possible activation functions for each node in the network
        self.nodes = PriorityQueue() #List of nodes in the network
        self.connections = [] #List of connections between nodes in the network
        self.innovation_counter = 0 #Innovation counter for adding new connections to the network
        self.output = 0
    
    def set_initial_graph(self):
        #TODO Set initial graph state
        pass
    
    def run(self, x, y, z) -> float:
        """
        Method to run the CPPN with given input paramaters

        :param x: x coordinate 
        :param y: y coordinate
        :param z: z coordinate
        """

        #Passes the input values into each input node in the network
        for node in self.nodes:
            if node.type == NodeType.INPUT:
                node.add_input(x)
                node.add_input(y)
                node.add_input(z)
                node.activate() #Activates each input node in the network after passing input paramaters
        
        #TODO activate nodes level by level until output produced

    def add_node(self, node) -> None:
        """
        Method to add a node to the CPPN

        :param node: node to be added to the CPPN
        """
        self.nodes.put(node) #Adds node to the list of nodes in the CPPN
    
    def reset(self) -> None:
        """
        Clears input and output values of each node in the network
        """
        for node in self.nodes: #Clears individual nodes I/O
            node.inputs = []
            node.output = 0
        self.output = 0 #Clears CPPN output value
    
    def create_connection(self, out, input, weight) -> None:
        """
        Method to create a connection between two nodes
        with a given weight

        :param out:
        :param input:
        :param weight:
        """
        new_connection = self.Connection(out, input, weight, self.innovation_counter) #Creates a new connection
        self.innovation_counter+=1 #Adds one to the innovation counter of the CPPN
        self.connections.append(new_connection) #Adds the new connection to the list of connections in the CPPN
    
    class Node:
        """
        Class defining a node in a compositional pattern-producing network
        """
        def __init__(self, activation_function, type, level, outer_cppn) -> None:
            """
            
            """
            self.inputs = [] #Input values passed into the node
            self.activation_function = activation_function #Activation function of the node
            self.type = type #Type of node (Input, Hidden, Output)
            self.output = 0 #Initilises the node output to 0
            self.level = level #Level the node is on in the network
            self.outer = outer_cppn
    
        def set_activation_function(self, activation_function) -> None:
            """
            Function to set the activation function of a node

            :param activation_function: activation function for the node to use
            """
            self.activation_function = activation_function
    
        def add_input(self, value) -> None:
            """
            Function to add a value to the input values into a node

            :param value: input value
            """
            self.inputs.append(value) #Input value added to the list of inputs to the node
    
        def activate(self) -> None:
            """
            Function to sum the input values into the node and 
            pass the total into the nodes activation function
            """
            total = 0
            for value in self.inputs:
                total += value #Sums the input values
        
            self.output = self.activation_function(total) #Sets the output value to the activation function applied on the summation of input values

            #TODO Check for connections to other nodes and feed (output * weight) to that node
            for connection in self.outer.connections:
                if connection.out is self:
                    connection.input.add_input(self.output * connection.weight)
        
        def __lt__(self, other):
            """

            """
            return self.level < other.level

    class Connection:
        """
        Class defining a connection between two nodes in a CPPN network
        """
        def __init__(self, out, input, weight, innov) -> None:
            """
            
            """
            self.out = out
            self.input = input
            self.weight = weight
            self.historical_marking = innov
            self.enabled = True
        
        def set_enabled(self, option) -> None:
            self.enabled = option
    
    
if __name__ == "__main__":
    """
    ******************
    DELETE FOR RELEASE
    TESTING ZONE
    ******************
    """
    a = CPPN()

    b = a.Node(sigmoid, NodeType.INPUT, 0, a)
    x = a.Node(symmetric, NodeType.INPUT, 0, a)
    c = a.Node(symmetric, NodeType.OUTPUT, 1, a)
    d = a.Node(identity, NodeType.OUTPUT, 1, a)
    i = a.Node(gaussian, NodeType.INPUT, 0, a)

    a.create_connection(b, c, 0.5)
    a.create_connection(b, d, 0.5)
    a.create_connection(x, c, 0.29139)

    i.add_input(0.1312)
    i.activate()

    print(i.output)

    b.add_input(1)
    x.add_input(-0.3)

    b.activate()
    x.activate()

    print(x.output)

    c.activate()
    d.activate()
    print(c.output)
    print(d.output)
    