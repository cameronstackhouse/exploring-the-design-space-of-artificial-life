from typing import Callable
from random import choice
from enum import Enum
import multiprocessing as mp
import numpy as np
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
    MATERIAL_OUTPUT = 3
    PRESENCE_OUTPUT = 4

class Node:
    """
    Class defining a node in a compositional pattern-producing network
    """
    def __init__(self, activation_function, type, level, outer_cppn) -> None:
        """
            
        """
        #TODO ADD NAME TO NODE
        self.inputs = [] #Input values passed into the node
        self.activation_function = activation_function #Activation function of the node
        self.type = type #Type of node (Input, Hidden, Output)
        self.output = None #Initilises the node output to none
        self.level = level #Level the node is on in the network
        self.outer = outer_cppn
        outer_cppn.add_node(self) #Adds the node to the CPPNs list of nodes
    
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
        total = 0 #Summation of input values
        num_connections_in = 0 #Number of enabled connections into the node

        #Iterates through the list of connections checking for connections into the node
        for connection in self.outer.connections:
            if self is connection.input and connection.enabled:
                num_connections_in+=1 #If the connection is a connection into the node, the number of connections counter is incremented
        
        #Checks if the number of conections into the node is the same as the number of inputs the node currently has
        if num_connections_in != len(self.inputs):
            for conection in self.outer.connections:
                #Activates the output node that has not been evaluated yet to provide the current node its input
                if self is conection.input and connection.enabled and connection.out.output is None:
                    conection.out.activate() #Activates the node that hasn't been activated yet

        for value in self.inputs:
            total += value #Sums the input values
        
        self.output = self.activation_function(total) #Sets the output value to the activation function applied on the summation of input values

        #Check for connections to other nodes and feed (output * weight) to that node
        for connection in self.outer.connections:
            if connection.out is self and connection.enabled:
                connection.input.add_input(self.output * connection.weight)
        
        #If the node is a presence output node then update the CPPN presence output value
        if self.type == NodeType.PRESENCE_OUTPUT:
            self.outer.presence = self.output
        
        #If the node is a material output node then update the CPPN material output value
        if self.type == NodeType.MATERIAL_OUTPUT:
            self.outer.material = self.output

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
        self.innovation_counter = 0 #Innovation counter for adding new connections to the network using NEAT
        self.material = None #Output indicating what type of material is present at a given location
        self.presence = None #Output indicating if material is present at a given location
    
    def set_initial_graph(self):
        #TODO Set initial graph state
        #Multiple Input nodes
        #Select 2 random activation functions for each output nodes
        #Connect the input function with the two output nodes 
        pass
    
    def run(self, i, j, k) -> None:
        """
        Method to run the CPPN with given input paramaters

        :param inputs: list of inputs passed into the CPPN
        :rtype: float
        :return: 
        """
        #TODO Add description
        #TODO Calculate d
        #TODO Change the bit bellow, need multiple input nodes w different inputs
        #Passes the input values into each input node in the network
        for node in self.nodes:
            if node.type is NodeType.INPUT:
                node.add_input(i)
                node.add_input(j)
                node.add_input(k)
                node.activate() #Activates each input node in the network after passing input paramaters
        
        #TODO Add comments
        for node in self.nodes:
            if node.type is not NodeType.INPUT:
                node.activate()


    def add_node(self, node) -> None:
        """
        Method to add a node to the CPPN

        :param node: node to be added to the CPPN
        """
        self.nodes.append(node) #Adds node to the list of nodes in the CPPN
    
    def reset(self) -> None:
        """
        Clears input and output values of each node in the network
        """
        for node in self.nodes: #Clears individual nodes I/O
            node.inputs = []
            node.output = None
        #Clears CPPN output values
        self.material = None
        self.presence = None 
    
    def create_connection(self, out, input, weight) -> None:
        """
        Method to create a connection between two nodes
        with a given weight

        :param out:
        :param input:
        :param weight:
        """
        #TODO Change to only add connection if on different layers
        new_connection = self.Connection(out, input, weight, self.innovation_counter) #Creates a new connection
        self.innovation_counter+=1 #Adds one to the innovation counter of the CPPN
        self.connections.append(new_connection) #Adds the new connection to the list of connections in the CPPN
    
    def to_phenotype(self):
        """
        Function to pass each point in a 8x8x7 3D design space into the
        CPPN to be mapped to a phenotype. The output at each point
        indicates if there is material at that point and, if so, what
        type of material it is (skin cell or cardiac cell)
        """
        #TODO Add ability to change the size of the 3D coordinate space (Use JSON settings file)
        results = np.zeros((8, 8, 7)) #Empty numpy array to store material results at each point
        
        #TODO Ensure this actually works with parallel processing
        try:
            pool = mp.Pool(mp.cpu_count()) #Initilises multiprocessing pool

            #Passes in every point in the 3D design space into the run function for the CPPN and then uses that data to determine what material is in each location
            #Produces a 3D numpy array modelling the 3D microorganism, with an integer at each point in the design space indicating material type/presence
            results = [pool.apply(self.material_produced(self.run), args=(i, j, k)) for i in range(8) for j in range(8) for k in range(7)]
        finally:
            #Closes multiprocessing pool
            pool.close()
            pool.join()

        return results
    
    def material_produced(result) -> int:
        """
        Function to convert a tuple result (produced from a CPPN
        when a coordinate point is passed into it) into an integer
        indicating what type of material exists at that location
        """
        #TODO
        pass

    def has_cycles(self) -> bool:
        #TODO
        pass
    
    def valid(self) -> bool:
        """
        Method to determine if nodes and connections in a CPPN topology are valid
        (At least one input node and two output nodes, one for 
        indicating presence of material and one for type of material)

        :rtype: bool
        :return: boolean indicating if the CPPN topology is valid
        """
        #TODO Add comments
        #TODO change input validation, should be 4 input nodes
        #Checks if the nodes are valid
        num_inputs = 0
        num_mat_out = 0
        num_presence_out = 0
        for node in self.nodes:
            if node.type is NodeType.INPUT:
                num_inputs+=1
            elif node.type is NodeType.MATERIAL_OUTPUT:
                num_mat_out+=1
            elif node.type is NodeType.PRESENCE_OUTPUT:
                num_presence_out+=1
        
        if num_inputs <= 0 or num_mat_out != 1 or num_presence_out != 1 or (num_presence_out + num_mat_out != 2):
            return False
        
        #Check if connections between nodes are valid
        #TODO

        if self.has_cycles():
            return False
        
        return True

    class Connection:
        """
        Class defining a connection between two nodes in a CPPN network
        """
        def __init__(self, out, input, weight, innov) -> None:
            """
            
            """
            #TODO Add description
            self.out = out
            self.input = input
            self.weight = weight
            self.historical_marking = innov
            self.enabled = True
        
        def set_enabled(self, option) -> None:
            """
            
            """
            #TODO Add description
            self.enabled = option
        
        def set_weight(self, value) -> None:
            """
            
            """
            #TODO Add description
            self.weight = value
    
if __name__ == "__main__":
    """
    ******************
    DELETE FOR RELEASE
    TESTING ZONE
    ******************
    """
    a = CPPN()

    b = Node(sigmoid, NodeType.INPUT, 0, a)
    x = Node(symmetric, NodeType.INPUT, 0, a)
    c = Node(symmetric, NodeType.PRESENCE_OUTPUT, 1, a)
    d = Node(identity, NodeType.MATERIAL_OUTPUT, 1, a)
  
    a.create_connection(b, c, 0.5)
    a.create_connection(b, d, 0.5)
    a.create_connection(x, c, 0.29139)

    b.add_input(1)
    x.add_input(1)

    b.activate()
    x.activate()

    c.activate()
    d.activate()
    print(c.output)
    print(d.output)

    print(a.valid())

    a.reset()

    a.run([1])

    for node in a.nodes:
        print(len(node.inputs))

    print(a.material)
    print(a.presence)