from typing import Callable
from random import choice
from enum import Enum
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
    def __init__(self, activation_function, type, level, outer_cppn, layer) -> None:
        """
            
        """
        self.inputs = [] #Input values passed into the node
        self.activation_function = activation_function #Activation function of the node
        self.type = type #Type of node (Input, Hidden, Output)
        self.output = 0 #Initilises the node output to 0
        self.level = level #Level the node is on in the network
        self.outer = outer_cppn
        outer_cppn.add_node(self, layer) #Adds the node to the CPPNs list of nodes
        #TODO Add comments
    
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
        
        if self.type == NodeType.MATERIAL_OUTPUT:
            self.outer.material = self.output
        
        if self.type == NodeType.PRESENCE_OUTPUT:
            self.outer.presence = self.output

        #Check for connections to other nodes and feed (output * weight) to that node
        for connection in self.outer.connections:
            if connection.out is self and connection.enabled:
                connection.input.add_input(self.output * connection.weight)
        #TODO Add comments
        #TODO If node is output node then send to CNN output

class Layer:
    """
    Layer in a neural network
    """
    def __init__(self) -> None:
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)
    
    def remove(self, index):
        self.nodes.remove(index)

class CPPN:
    """
    Class defining a compositional pattern-producing network made of
    interconnected nodes with varying activation functions
    """
    def __init__(self) -> None:
        """
        
        """
        self.activation_functions = [sigmoid, periodic, identity, gaussian, repeat_asym, absolute, inverse, symmetric] #List of possible activation functions for each node in the network
        self.layers = [] #List of layers of nodes in the network
        self.connections = [] #List of connections between nodes in the network
        self.innovation_counter = 0 #Innovation counter for adding new connections to the network
        self.material = 0 #Output indicating what type of material is present at a given location
        self.presence = 0 #Output indicating if material is present at a given location
    
    def set_initial_graph(self):
        #TODO Set initial graph state
        #Select random activation function for use for an input node
        #Select 2 random activation functions for each output nodes
        #Connect the input function with the two output nodes
        layer_zero = Layer()
        layer_one = Layer()

    
    def add_layer(self):
        """
        
        """
        #TODO Add description
        self.layers.append(Layer())

    def add_node(self, node, layer):
        """
        
        """

        #TODO Add description
        self.layers[layer].nodes.append(node)
    
    def run(self, inputs) -> float:
        """
        Method to run the CPPN with given input paramaters

        :param inputs: list of inputs passed into the CPPN
        :rtype: float
        :return: 
        """
        #TODO Change input method??

        #Passes the input values into each input node in the network (Layer 0)
        for node in self.layers[0].nodes:
            for input in inputs:
                node.add_input(input)
            node.activate()
        
        #Activates nodes level by level until all nodes have been activated produced
        for i in range(1, len(self.layers)): #Loops through all non input layers
            for node in self.layers[i].nodes: #Iterates through nodes in a given layer
                node.activate() #Activates the node

    
    def reset(self) -> None:
        """
        Clears input and output values of each node in the network
        """
        for layer in self.layers:
            for node in layer.nodes:
                node.inputs = []
                node.output = 0
        
        self.presence = None
        self.material = None
        
    def create_connection(self, out, input, weight) -> None:
        """
        Method to create a connection between two nodes
        with a given weight

        :param out:
        :param input:
        :param weight:
        """
        #TODO Change to only add connection if on different layers and don't already have connection
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
        results = np.zeros((8, 8, 7))
        for i in range(8):
            for j in range(8):
                for k in range(7):
                    #TODO Pass in d (distance from middle)
                    cppn_result = self.run([i, j, k]) 
                    material = self.material_produced(cppn_result)
                    results[i,j,k] = material

        return results
    
    def material_produced(result) -> int:
        """
        Function to convert a tuple result (produced from a CPPN
        when a coordinate point is passed into it) into an integer
        indicating what type of material exists at that location
        """
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
        #Checks if the nodes are valid
        #TODO Change code to be valid with layers
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
        return True

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
            #TODO Add description
        
        def set_enabled(self, option) -> None:
            """
            
            """
            self.enabled = option
            #TODO Add description
        
        def set_weight(self, value) -> None:
            """
            
            """
            self.weight = value
            #TODO Add description
    
if __name__ == "__main__":
    """
    ******************
    DELETE FOR RELEASE
    TESTING ZONE
    ******************
    """
    a = CPPN()

    a.add_layer()
    a.add_layer()

    b = Node(sigmoid, NodeType.INPUT, 0, a, 0)
    x = Node(symmetric, NodeType.INPUT, 0, a, 0)
    c = Node(symmetric, NodeType.PRESENCE_OUTPUT, 1, a, 1)
    d = Node(identity, NodeType.MATERIAL_OUTPUT, 1, a, 1)
    i = Node(gaussian, NodeType.INPUT, 0, a, 0)
  
    a.create_connection(b, c, 0.5)
    a.create_connection(x, c, 0.29139)
    a.create_connection(x, d, 1)

    i.add_input(1)
    b.add_input(1)
    x.add_input(1)

    i.activate()
    b.activate()
    x.activate()

    c.activate()
    d.activate()
    print(f"Presence: {c.output}")
    print(f"Material: {d.output}")

    a.reset()

    a.run([1])

    print("GAP")

    print(a.material)
    print(a.presence)
    