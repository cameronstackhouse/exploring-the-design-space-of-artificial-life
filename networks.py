from typing import Callable
from random import choice, uniform
from enum import Enum
import multiprocessing as mp
import numpy as np
from utilities.activation_functions import sigmoid, neg_abs, neg_square, sqrt_abs, neg_sqrt_abs #Imports all activation functions

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
    INPUT_I = 5
    INPUT_J = 6
    INPUT_K = 7
    INPUT_D = 8

class Node:
    """
    Class defining a node in a compositional pattern-producing network
    """
    def __init__(self, activation_function, type, outer_cppn) -> None:
        """
            
        """
        #TODO Add description
        self.inputs = [] #Input values passed into the node
        self.activation_function = activation_function #Activation function of the node
        self.type = type #Type of node (Input, Hidden, Output)
        self.output = None #Initilises the node output to none
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
    
    def set_inputs(self, inputs) -> None:
        """
        
        """
        self.inputs = inputs
    
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
        

        #TODO FIX THIS! EXCEEDS RECURSION DEPTH (MAYBE???) (DYNAMIC PROGRAMMING?)
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
    def __init__(self, xyz_size) -> None:
        """
        Function to initilise an empty CPPN
        """
        self.activation_functions = [np.sin, np.abs, neg_abs, np.square, neg_square, sqrt_abs, neg_sqrt_abs] #List of possible activation functions for each node in the network
        self.nodes = [] #List of nodes in the network
        self.connections = [] #List of connections between nodes in the network
        self.innovation_counter = 0 #Innovation counter for adding new connections to the network using NEAT
        self.material = None #Output indicating what type of material is present at a given location
        self.presence = None #Output indicating if material is present at a given location
        self.x_inputs = []
        self.y_inputs = []
        self.z_inputs = []
        self.d_inputs = []
        self.b_inputs = []
        self.set_initial_graph(xyz_size) #Sets the initial graph
    
    def set_initial_graph(self, xyz_size):
        """
        Function to set the initial graph of the CPPN
        with the correct input and output nodes for each 
        paramater in a 3D coordinate space (i, j, k, and distance from middle)

        Also sets the two output nodes, one to indicate the presence of material
        at a given coordinate and one to indicate the material of that node
        """


        #TODO NORMALIZE INPUTS TO NN (Make between 0 and 1). SET THESE IN THE MINIMAL GRAPH
        #TODO Calculate d (distance from centre)
        #TODO Add b as an input value
        #Creates an input node for each paramater: i, j, k, and d
        for type in [NodeType.INPUT_I, NodeType.INPUT_J, NodeType.INPUT_K, NodeType.INPUT_D]:
            activation_function = choice(self.activation_functions) #Chooses a random activation function
            Node(activation_function, type, self) #Creates the new node, automatically adding it to the CPPN

        #Creates an output node for material and presence
        material = Node(sigmoid, NodeType.MATERIAL_OUTPUT, self)
        presence = Node(sigmoid, NodeType.PRESENCE_OUTPUT, self)

        #Connects the input functions with the two output nodes 
        for node in self.nodes:
            if node.type is not NodeType.MATERIAL_OUTPUT and node.type is not NodeType.PRESENCE_OUTPUT: #If the node isn't an output node
                #Connect the node to the two output nodes
                self.create_connection(node, material, uniform(0,1))
                self.create_connection(node, presence, uniform(0,1))
    
    def run(self, i, j, k) -> None:
        """
        Method to run the CPPN with given input paramaters

        :param inputs: list of inputs passed into the CPPN
        :rtype: float
        :return: 
        """
        #TODO Add description
        #TODO Change to provid single input (1 to 8*8*7)
        #TODO ADD b
        d=0

        #Passes the input values into each input node in the network
        for node in self.nodes:
            if node.type is NodeType.INPUT_I:
                node.add_input(i)
                node.activate() #Activates the node
            elif node.type is NodeType.INPUT_J:
                node.add_input(j)
                node.activate() #Activates the node
            elif node.type is NodeType.INPUT_K:
                node.add_input(k)
                node.activate() #Activates the node
            elif node.type is NodeType.INPUT_D:
                node.add_input(d)
                node.activate() #Activates the node
        
        #TODO Add comments
        for node in self.nodes:
            if node.type is not (NodeType.INPUT_J or NodeType.INPUT_I or NodeType.INPUT_K or NodeType.INPUT_D):
                node.activate()


    def add_node(self, node) -> None:
        """
        Method to add a node to the CPPN

        :param node: node to be added to the CPPN
        """
        self.nodes.append(node) #Adds node to the list of nodes in the CPPN
    
    def add_node_between(self, connection) -> bool:
        #TODO
        #Check if new topology is valid
        pass
    
    def reset(self) -> None:
        """
        Clears input and output values of each node in the network
        """
        #TODO After normalized inputs DO NOT clear input nodes inputs
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
        #TODO Description
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
            results = [pool.apply(self.material_produced(self.run), args=[i, j, k]) for i in range(8) for j in range(8) for k in range(7)]
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
        #TODO CHECK FOR 1 OF EACH INPUT NODE (I, J, K, D)
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
        
        if num_inputs != 4 or num_mat_out != 1 or num_presence_out != 1 or (num_presence_out + num_mat_out != 2):
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

def normalize(x):
    """
    
    """
    #TODO Add description
    return ((x - np.min(x)) / (np.max(x) - np.min(x)))
    
if __name__ == "__main__":
    """
    ******************
    DELETE FOR RELEASE
    TESTING ZONE
    ******************
    """
    a = CPPN(8*8*7)

    # b = Node(sigmoid, NodeType.INPUT, a)
    # x = Node(symmetric, NodeType.INPUT, a)
    # c = Node(symmetric, NodeType.PRESENCE_OUTPUT, a)
    # d = Node(identity, NodeType.MATERIAL_OUTPUT, a)
  
    # a.create_connection(b, c, 0.5)
    # a.create_connection(b, d, 0.5)
    # a.create_connection(x, c, 0.29139)

    # b.add_input(1)
    # x.add_input(1)

    # b.activate()
    # x.activate()

    # c.activate()
    # d.activate()
    # print(c.output)
    # print(d.output)

    # print(a.valid())

    # a.reset()
    

    a.run(20, 3, 1)

    for node in a.nodes:
        print(len(node.inputs))

    print(f"Material: {a.material}")
    print(f"Presence: {a.presence}")

    for connection in a.connections:
    
        print(f"Out: {connection.out.activation_function} ({connection.out.type}) Into: {connection.input.activation_function} ({connection.input.type})")
