"""
Module defining components for the creation of functioning
compositional pattern-producing networks
"""

from random import choice, uniform
from enum import Enum
import multiprocessing as mp
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from tools.activation_functions import sigmoid, neg_abs, neg_square, sqrt_abs, neg_sqrt_abs, normalize #Imports all activation functions

class NodeType(Enum):
    """
    Class defining the possible node types in the CPPN
    """
    HIDDEN = 1
    MATERIAL_OUTPUT = 2
    PRESENCE_OUTPUT = 3
    INPUT_X = 4
    INPUT_Y = 5
    INPUT_Z = 6
    INPUT_D = 7
    INPUT_B = 8

class Node:
    """
    Class defining a node in a compositional pattern-producing network
    """
    def __init__(self, activation_function, type, outer_cppn, layer) -> None:
        """
        Initilization method for creating a CPPN node

        :param activation_function: activation function used by the node
        :param type: type of node (INPUT, HIDDEN, OUTPUT)
        :param outer_cppn: CPPN which the node belongs to
        """
        self.inputs = [] #Input values passed into the node
        self.activation_function = activation_function #Activation function of the node
        self.type = type #Type of node (Input, Hidden, Output)
        self.output = None #Initilises the node output to none
        self.outer = outer_cppn
        self.previous_out = None
        self.previous_in = None
        outer_cppn.add_node(self, layer) #Adds the node to the CPPNs list of nodes
    
    def set_activation_function(self, activation_function) -> None:
        """
        Function to set the activation function of a node

        :param activation_function: activation function for the node to use
        """
        self.activation_function = activation_function
    
    def add_input(self, value: float) -> None:
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
    interconnected nodes with varying activation functions for use in the
    design of reconfigurable organisms
    """
    def __init__(self, xyz_size: List) -> None:
        """
        Function to initilise a basic CPPN for designing reconfigurable organsims,
        setting an initial graph (5 input nodes fully connected to 2 output nodes)
        and setting the x, y, z, d, and b inputs needed for the CPPN to fully generate
        a design

        :param xyz_size: volume of the coordinate space for design
        """
        #TODO Add ability to change the size of the 3D coordinate space (Use JSON settings file)
        #TODO Change node comments
        self.activation_functions = [np.sin, np.abs, neg_abs, np.square, neg_square, sqrt_abs, neg_sqrt_abs] #List of possible activation functions for each node in the network
        self.nodes = [[], []] #List of nodes in the network
        self.connections = [] #List of connections between nodes in the network
        self.innovation_counter = 0 #Innovation counter for adding new connections to the network using NEAT
        self.material = None #Output indicating what type of material is present at a given location
        self.presence = None #Output indicating if material is present at a given location
        self.phenotype = None
        #Lists of inputs for each input node
        self.x_inputs = []
        self.y_inputs = []
        self.z_inputs = []
        self.d_inputs = []
        self.b_inputs = []
        self.fitness = None
        self.xyz_size = xyz_size #Dimentions of the design space
        self.temp_mark = []
        self.unmarked = []
        self.set_initial_graph() #Sets the initial graph
    
    def set_input_states(self) -> None:
        """
        Function to set the input states of the CPPN given the
        dimensions of the design space
        """
        #Creates empty numpy arrays for each x, y, and z inputs, each the size of the volume of the design space
        x_inputs = np.zeros(self.xyz_size)
        y_inputs = np.zeros(self.xyz_size)
        z_inputs = np.zeros(self.xyz_size)

        #Populates each coordinate input array with the correct values
        for x in range(self.xyz_size[0]):
            for y in range(self.xyz_size[1]):
                for z in range(self.xyz_size[2]):
                    x_inputs[x, y, z] = x
                    y_inputs[x, y, z] = y
                    z_inputs[x, y, z] = z
        
        #Normalizes the inputs to make them all between -1 and 1
        x_inputs = normalize(x_inputs)
        y_inputs = normalize(y_inputs)
        z_inputs = normalize(z_inputs)

        #Creates the d input array, calculating the distance each point is away from the centre
        d_inputs = normalize(np.power(np.power(x_inputs, 2) + np.power(y_inputs, 2) + np.power(z_inputs, 2), 0.5))

        #Creates the b input array, which is just a numpy array of ones
        b_inputs = np.ones(self.xyz_size)

        #Sets all inputs and flattens them into 1D arrays
        self.x_inputs = x_inputs.flatten()
        self.y_inputs = y_inputs.flatten()
        self.z_inputs = z_inputs.flatten()
        self.d_inputs = d_inputs.flatten()
        self.b_inputs = b_inputs.flatten()
    
    def set_initial_graph(self) -> None:
        """
        Function to set the initial graph of the CPPN
        with the correct input and output nodes for each 
        paramater in a 3D coordinate space (i, j, k, and distance from middle)

        Also sets the two output nodes, one to indicate the presence of material
        at a given coordinate and one to indicate the material of that node
        """

        self.set_input_states()
        #Creates an input node for each paramater: i, j, k, d, and b
        for type in [NodeType.INPUT_X, NodeType.INPUT_Y, NodeType.INPUT_Z, NodeType.INPUT_D, NodeType.INPUT_B]:
            activation_function = choice(self.activation_functions) #Chooses a random activation function
            Node(activation_function, type, self, 0) #Creates the new node, automatically adding it to the first layer in the CPPN

        #Creates an output node for material and presence and adds both to the output layer
        material = Node(sigmoid, NodeType.MATERIAL_OUTPUT, self, 1)
        presence = Node(sigmoid, NodeType.PRESENCE_OUTPUT, self, 1)

        #Connects the input nodes with the two output nodes
        for node in self.nodes[0]:
            #Connect the node to the two output nodes
            self.create_connection(node, material, uniform(0,1))
            self.create_connection(node, presence, uniform(0,1))
    
    def run(self, pixel: int) -> int:
        """
        Method to run the CPPN with given input paramaters,
        updating the CPPN with two output values, one indicating
        if material exists at a given voxel and one indicating
        what that material at that voxel is

        :param pixel: the voxel number to be computed
        """

        self.reset() #Clears already existing values in CPPN

        #Passes the correct input values into each input node in the network at the given point
        for node in self.nodes[0]:
            if node.type is NodeType.INPUT_X:
                node.add_input(self.x_inputs[pixel])
                node.activate() #Activates the node
            elif node.type is NodeType.INPUT_Y:
                node.add_input(self.y_inputs[pixel])
                node.activate() #Activates the node
            elif node.type is NodeType.INPUT_Z:
                node.add_input(self.z_inputs[pixel])
                node.activate() #Activates the node
            elif node.type is NodeType.INPUT_D:
                node.add_input(self.d_inputs[pixel])
                node.activate() #Activates the node
            elif node.type is NodeType.INPUT_B:
                node.add_input(self.b_inputs[pixel])
                node.activate() #Activates the node
        
        for layer in self.nodes[1:]: #Iterates through all nodes and activates all non input nodes (as they have already been activated)
            for node in layer:
                node.activate()
        
        return self.material_produced() #Returns an integer indicating the material at that voxel

    def add_node(self, node: Node, layer: int) -> None:
        """
        Method to add a node to the CPPN

        :param node: node to be added to the CPPN
        :param layer: layer of the CPPN to add the node into
        """
        self.nodes[layer].append(node) #Adds node to the list of nodes in the CPPN
    
    def remove_node(self, node: Node):
        """
        
        """
        #TODO Add comments
        #Checks if the node is a hidden node, only hidden nodes can be deleted
        if node.type is NodeType.HIDDEN:
            for n, layer in enumerate(self.nodes):
                if node in layer:
                    layer.remove(node)
                    
                    if len(layer) == 0: #Checks if layer is now empty, if so the layer is removed
                        self.nodes.pop(n)
                    break

    def reset(self) -> None:
        """
        Clears input and output values of each node in the network
        """
        for layer in self.nodes: #Clears individual nodes I/O
            for node in layer:
                node.inputs = []
                node.output = None
        #Clears CPPN output values
        self.material = None
        self.presence = None 
    
    def create_connection(self, out: Node, input: Node, weight: float) -> bool:
        """
        Method to create a connection between two nodes
        with a given weight

        :param out: output node
        :param input: input node
        :param weight: weight associated with the connection
        """
        for connection in self.connections:
            if connection.out is out and connection.input is input:
                return False

        new_connection = self.Connection(out, input, weight, self.innovation_counter) #Creates a new connection
        self.connections.append(new_connection) #Adds the new connection to the list of connections in the CPPN
        # if self.has_cycles(): #Checks if the CPPN with the new connection has cycles
        #     self.connections.remove(new_connection) #If so remove the new connection
        #     return False
        self.innovation_counter+=1 #Adds one to the innovation counter of the CPPN

        return True
    
    def to_phenotype(self) -> np.array:
        """
        Function to pass each point in an 3D design space into the
        CPPN to be mapped to a phenotype. The output at each point
        indicates if there is material at that point and, if so, what
        type of material it is (skin cell or cardiac cell)
        """
        results = np.zeros(self.xyz_size) #Empty numpy array to store material results at each point
        
        try:
            #Gets the volume of the coordinate space
            size = 1
            for number in self.xyz_size:
                size*=number

            pool = mp.Pool(mp.cpu_count()) #Initilises multiprocessing pool

            #Passes in every point in the 3D design space into the run function for the CPPN and then uses that data to determine what material is in each location
            #Produces a 3D numpy array modelling the 3D microorganism, with an integer at each point in the design space indicating material type/presence
            results = np.array([pool.apply(self.run, args=[i]) for i in range(size)])
        finally:
            #Closes multiprocessing pool
            pool.close()
            pool.join()
        
        self.phenotype = results

        return results
    
    def material_produced(self) -> int:
        """
        Function to convert a tuple result (produced from a CPPN
        when a coordinate point is passed into it) into an integer
        indicating what type of material exists at that location
        """
        #TODO Add comments
        #TODO Ask chico about this!
        presence = self.presence
        material = self.material
        if presence < 0.3:
            return 0
        elif material < 0.7:
            return 1
        else:
            return 2
        
    def num_activation_functions(self) -> dict:
        """
        Function to return the quantity of each different activation
        function in the network

        :rtype: dict
        :return: dictionary containing activation functions as keys and the quantity of each
        """
        functions = dict()
        #Iterates through all nodes in a network, adding their activation function to a dictionary or incrementing their counter
        for layer in self.nodes:
            for node in layer:
                if node.activation_function in functions:
                    functions[node.activation_function] = functions[node.activation_function]+1
                else:
                    functions[node.activation_function] = 1
        
        return functions

    def num_cells(self) -> dict:
        """
        Function to return the number of the three different types of cells in the 
        phenotype

        :rtype: dict
        :return dictionary containing number of skin and muscle cells
        """
        none = 0
        skin = 0
        muscle = 0
        #Iterates through phenotype cells and increments the associated cell counter
        for cell in self.phenotype:
            if cell == 0:
                none+=1
            elif cell == 1:
                skin+=1
            elif cell == 2:
                muscle+=1
        
        return {"none": none, "skin": skin, "muscle": muscle}
    
    def num_cells_at_location(self):
        #TODO maybe count of cells within top left, top right... 
        pass

    def phenotype_symmetry(self):
        #TODO Determines the "symmetry" of the phenotype on each axis
        pass

    def valid_connections(self) -> bool:
        """
        Function to determine if a CPPN has valid connections by
        checking that every connection has the out node in
        a lower layer to the input node

        :rtype: bool
        :return: boolean value indicating if the CPPN has valid connections
        """
        #Iterates through all enabled connections
        for connection in self.connections:
            if connection.enabled:
                out = connection.out
                input = connection.input

                out_layer = 0
                input_layer = 0

                #Finds the layer index that the out and input nodes are on
                for n, layer in enumerate(self.nodes):
                    if out in layer:
                        out_layer = n
                    elif input in layer:
                        input_layer = n

                #Checks to ensure the layer index of input is lower than out
                if out_layer >= input_layer:
                    return False #If not return false, as there is an invalid connection 
        
        return True #If all connections are valid return true

    class Connection:
        """
        Class defining a connection between two nodes in a CPPN network
        """
        def __init__(self, out: Node, input: Node, weight: float, innov: int) -> None:
            """
            Initilization method to create a basic connection
            between two nodes

            :param out: output node
            :param input: input node
            :param weight: weight associated with the connection
            :param innov: historical marking for the connection
            """
            self.out = out 
            self.input = input
            self.weight = weight
            self.historical_marking = innov
            self.enabled = True #Automatically enables the connection
        
        def set_enabled(self, option: bool) -> None:
            """
            Method to set if a connection is enabled

            :param option: boolean value indicating if the connection is enabled or disabled
            """
            self.enabled = option
        
        def set_weight(self, value: float) -> None:
            """
            Method to set the weight associated with a connection

            :param value: weight of the connection
            """
            self.weight = value

    
if __name__ == "__main__":
    """
    ******************
    DELETE FOR RELEASE
    TESTING ZONE
    ******************
    """
    a = CPPN([8,8,7])

    print(a.valid_connections())
   
    b = a.to_phenotype()

    newarr = b.reshape(8,8,7)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    data = newarr
    z,x,y = data.nonzero()

    ax.scatter(x, y, z, cmap='coolwarm', alpha=1)
    plt.show()

    nc = a.num_cells()
    na = a.num_activation_functions()

    for x in nc:
        print(f"{x}: {nc[x]}")
    
    for x in na:
        print(f"{x}: {na[x]}")
    