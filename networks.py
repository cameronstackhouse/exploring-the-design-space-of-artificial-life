"""
Module defining components for the creation of functioning
compositional pattern-producing networks
"""

import pickle
from copy import copy, deepcopy
from random import choice, random
from enum import Enum
import multiprocessing as mp
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from tools.activation_functions import sigmoid, neg_abs, neg_square, sqrt_abs, neg_sqrt_abs, normalize #Imports all activation functions
from tools.draw_cppn import draw_cppn

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
    def __init__(
        self, 
        activation_function, 
        type, 
        outer_cppn, 
        layer
        ) -> None:
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
        self.name = None
        self.layer = layer
        self.position = 0 # TODO MAKE SURE THIS WORKS!
        outer_cppn.add_node(self, layer) #Adds the node to the CPPNs list of nodes
    
    def set_activation_function(
        self, 
        activation_function
        ) -> None:
        """
        Function to set the activation function of a node

        :param activation_function: activation function for the node to use
        """
        self.activation_function = activation_function
    
    def add_input(
        self, 
        value: float
        ) -> None:
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

        if len(self.inputs) > 0: # Checks to make sure the number of inputs is greater than 0
            total = 0 #Summation of input values

            for value in self.inputs:
                total += value #Sums the input values
            
            if self.activation_function is not None:
                self.output = self.activation_function(total) #Sets the output value to the activation function applied on the summation of input values
            else:
                self.output = total

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
    innovation_counter = 0 #Innovation counter for adding new connections to the network using NEAT
    total_cons = [] #List of all connections created between nodes of all CPPNs 
    def __init__(self, xyz_size: List) -> None:
        """
        Function to initilise a basic CPPN for designing reconfigurable organsims,
        setting an initial graph (5 input nodes fully connected to 2 output nodes)
        and setting the x, y, z, d, and b inputs needed for the CPPN to fully generate
        a design

        :param xyz_size: list of length 3 indicating the size of each axis (x,y,z) in the design space
        """
        self.activation_functions = [np.sin, np.abs, neg_abs, np.square, neg_square, sqrt_abs, neg_sqrt_abs] #List of possible activation functions for each node in the network
        self.nodes = [[], []] #List of nodes in the network
        self.connections = [] #List of connections between nodes in the network
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
        self.movement = [] #List describing the movement of the xenobot in the x, y, and z direction
        self.xyz_size = xyz_size #Dimentions of the design space
        self.neighbours = [] #Neighbours in Genotype-Phenotype map
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
        at a given coordinate and one to indicate the material of that node.
        """
        self.set_input_states()
        #Creates an input node for each paramater: i, j, k, d, and b
        for type in [NodeType.INPUT_X, NodeType.INPUT_Y, NodeType.INPUT_Z, NodeType.INPUT_D, NodeType.INPUT_B]:
            activation_function = None
            Node(activation_function, type, self, 0) #Creates the new node, automatically adding it to the first layer in the CPPN

        #Creates an output node for material and presence and adds both to the output layer
        material = Node(sigmoid, NodeType.MATERIAL_OUTPUT, self, 1)
        presence = Node(sigmoid, NodeType.PRESENCE_OUTPUT, self, 1)

        #Connects the input nodes with the two output nodes
        for node in self.nodes[0]:
            #Connect the node to the two output nodes
            self.create_connection(node, material, random())
            self.create_connection(node, presence, random())

    def run(
        self, 
        pixel: int
        ) -> int:
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

    def add_node(
        self, 
        node: Node, 
        layer: int
        ) -> None:
        """
        Method to add a node to the CPPN

        :param node: node to be added to the CPPN
        :param layer: layer of the CPPN to add the node into
        """
        position = 0
        # TODO ensure this works: Just count where the node should be starting at first layer?
        #Finds the position as to where the new node is going to be
        for layer_pos in self.nodes[:layer+1]:
            position += len(layer_pos)
        
        node.position = position
        self.nodes[layer].append(node) #Adds node to the list of nodes in the CPPN

        #Updates all the positions of the nodes infront of the inserted node
        for layer in self.nodes[layer+1:]:
            for node in layer:
                node.position = position+1
                position+=1
        
        #TODO have we got to then update connections? Lets find out lads (Do not think so but we will make sure
        # )

    def remove_node(
        self, 
        node: Node
        ) -> None:
        """
        Function to remove a node from a CPPN

        :param node: Node to be removed
        """
        #TODO FINISH COMMENTS
        #TODO Make sure works too :)
        #Checks if the node is a hidden node, only hidden nodes can be deleted
        if node.type is NodeType.HIDDEN:
            for n, layer in enumerate(self.nodes): #Iterates through layers of CPPN
                if node in layer: #Checks if node is in the layer
                    node_pos = node.position
                    index = layer.index(node)
                    layer.remove(node) #Removes node from the layer
                    if len(layer) == 0: #Checks if layer is now empty, if so the layer is removed
                        self.nodes.pop(n) #Pops the layer
                    else:
                        #Updates node positions of nodes in the layer which the node has been popped from
                        for node in layer[index:]: 
                            node.position = node_pos
                            node_pos+=1
                    
                    for layer in self.nodes[n+1:]:
                        for node in layer:
                            node.position = node_pos
                            node_pos+=1
                    
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
    
    def create_connection(
        self, 
        out: Node, 
        input: Node, 
        weight: float
        ) -> bool:
        """
        Method to create a connection between two nodes
        with a given weight

        :param out: output node
        :param input: input node
        :param weight: weight associated with the connection
        """
        #Makes sure the connection does not exist in the topology
        for connection in self.connections:
            if connection.out is out and connection.input is input:
                return False

        #Check if a connection between two nodes in the same position exists in another topology
        exists = False
        for connection in CPPN.total_cons:
            if connection.out.position == out.position and connection.input.position == input.position: #Checks that the node positions match the connection node positions
                new_connection = self.Connection(out, input, weight, connection.historical_marking) #Creates a new connection using the existing historical marking (prevents historical marking explosion)
                self.connections.append(new_connection) #Adds the new connection to the list of connections in the CPPN
                exists = True
                break

        if not exists: #If there isn't an already existing matching connection in a different topology
            new_connection = self.Connection(out, input, weight, CPPN.innovation_counter) #Creates a new connection using innovation counter
            CPPN.total_cons.append(new_connection)
            self.connections.append(new_connection) #Adds the new connection to the list of connections in the CPPN
            CPPN.innovation_counter += 1 #Increments innovation counter

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

        :rtype: int
        :return: Value indicating material at a given location (0 for none, 1 for skin, and 2 for cardiac)
        """
        #TODO Ask chico about this!
        presence = self.presence #Gets presence output of CPPN
        material = self.material #Gets material output of CPPN
        if presence <= 0.2: #Checks if presence output is less than 0.5
            return 0 #If so there is no material in the location
        elif material < 0.5: #Checks if material output is less than 0.5
            return 1 #If so there is skin in the location
        else:
            return 2 #Else there is a cardiac cell in the location
        
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
    
    def connection_types(self) -> dict:
        """
        Function to get information about how many connections are
        enabled and disabled in the CPPN

        :rtype: dict
        :return: dictionary with two keys "enabled" and "disabled" both with counters as values
        """
        enabled_counter = 0
        disabled_counter = 0
        for connection in self.connections: #Iterates through connections
            if connection.enabled: #Checks if connection is enabled
                enabled_counter+=1
            else:
                disabled_counter+=1
        
        return {"enabled": enabled_counter, "disabled": disabled_counter}
    
    def has_cycles(self) -> bool:
        """
        Function determining if a CPPN contains cycles.

        :rtype: boolean
        :return: boolean describing if a graph has cycles
        """
        cycles = False
        
        # Iterates through all input nodes, checking for cycles at each
        for node in self.nodes[0]:
            visited = set() # Keeps a visited set
            stack = [node]

            while len(stack) != 0:
                current = stack.pop()
                if current in visited:
                    cycles = True
                    break
                else:
                    visited.add(current)
                    for connection in self.connections:
                        if connection.out is current and connection.enabled:
                            stack.append(connection.input)
            
            if cycles:
                break
                
        return cycles
        
    class Connection:
        """
        Class defining a connection between two nodes in a CPPN network
        """
        def __init__(
            self, 
            out: Node, 
            input: Node, 
            weight: float, 
            innov: int
            ) -> None:
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
        
        def set_enabled(
            self, 
            option: bool
            ) -> None:
            """
            Method to set if a connection is enabled

            :param option: boolean value indicating if the connection is enabled or disabled
            """
            self.enabled = option
        
        def set_weight(
            self, 
            value: float
            ) -> None:
            """
            Method to set the weight associated with a connection

            :param value: weight of the connection
            """
            self.weight = value

class PopulationContainer:
    def __init__(population, name, description):
        self.population = population
        self.name = name
        self.description = description
    
    def best(self) -> CPPN:
        best = None
        for cppn in self.population:
            if best is None or best.fitness < cppn.fitness:
                best = cppn
    
def save_population(population, filename):
    pickle.dump(population, open(filename, "wb"))

def load_population(filename) -> PopulationContainer:
    return pickle.load(open(filename, "rb"))

            
if __name__ == "__main__":
    """
    ******************
    DELETE FOR RELEASE
    TESTING ZONE
    ******************
    """
    a = CPPN([8,8,7])

    pickle.dump(a, open("test.p", "wb"))

    b = pickle.load(open("test.p", "rb"))


    # dec = input("Want to see it?")

    # if dec.lower() == "yes":

    #     draw_cppn(a, show_weights=True)

    #     print(a.valid_connections())
   
    #     b = a.to_phenotype()

    #     print(b)

    #     newarr = b.reshape(8,8,7)

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     data = newarr
    #     z,x,y = data.nonzero()

    #     ax.scatter(x, y, z, cmap='coolwarm', alpha=1)
    #     plt.show()
    