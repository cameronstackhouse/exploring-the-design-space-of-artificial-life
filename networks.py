from random import choice, uniform
from enum import Enum
import multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt
from utilities.activation_functions import sigmoid, neg_abs, neg_square, sqrt_abs, neg_sqrt_abs, normalize #Imports all activation functions

"""
Module defining components for the creation of functioning
compositional pattern-producing networks
"""

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
    def __init__(self, activation_function, type, outer_cppn) -> None:
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
    interconnected nodes with varying activation functions for use in the
    design of reconfigurable organisms
    """
    def __init__(self, xyz_size) -> None:
        """
        Function to initilise a basic CPPN for designing reconfigurable organsims,
        setting an initial graph (5 input nodes fully connected to 2 output nodes)
        and setting the x, y, z, d, and b inputs needed for the CPPN to fully generate
        a design

        :param xyz_size: volume of the coordinate space for design
        """
        #TODO Add ability to change the size of the 3D coordinate space (Use JSON settings file)
        self.activation_functions = [np.sin, np.abs, neg_abs, np.square, neg_square, sqrt_abs, neg_sqrt_abs] #List of possible activation functions for each node in the network
        self.nodes = [] #List of nodes in the network
        self.connections = [] #List of connections between nodes in the network
        self.innovation_counter = 0 #Innovation counter for adding new connections to the network using NEAT
        self.material = None #Output indicating what type of material is present at a given location
        self.presence = None #Output indicating if material is present at a given location
        #Lists of inputs for each input node
        self.x_inputs = []
        self.y_inputs = []
        self.z_inputs = []
        self.d_inputs = []
        self.b_inputs = []
        self.xyz_size = xyz_size #Dimentions of the design space
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
        for node in self.nodes:
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
        
        for node in self.nodes: #Iterates through all nodes and activates all non input nodes (as they have already been activated)
            if node.type is not (NodeType.INPUT_Y or NodeType.INPUT_X or NodeType.INPUT_Z or NodeType.INPUT_D or NodeType.INPUT_B):
                node.activate()
        
        return self.material_produced() #Returns an integer indicating the material at that voxel

    def add_node(self, node) -> None:
        """
        Method to add a node to the CPPN

        :param node: node to be added to the CPPN
        """
        self.nodes.append(node) #Adds node to the list of nodes in the CPPN
    
    def add_node_between(self, connection) -> bool:
        """
        
        """
        #TODO
        #Check if new topology is valid
        pass
    
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

        :param out: output node
        :param input: input node
        :param weight: weight associated with the connection
        """
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

        return results
    
    def material_produced(self) -> int:
        """
        Function to convert a tuple result (produced from a CPPN
        when a coordinate point is passed into it) into an integer
        indicating what type of material exists at that location
        """
        #TODO VASTLY NEEDS IMPROVING
        presence = self.presence
        material = self.material
        if presence < 0.25 or presence > 0.75:
            return 0
        elif material < 0.5:
            return 1
        else:
            return 2

    def has_cycles(self) -> bool:
        """
        Method to determine if a given CPPN has cycles
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
        #TODO Add comments
        #TODO CHECK FOR 1 OF EACH INPUT NODE (I, J, K, D, B)
        #Checks if the nodes are valid
        num_inputs = 0
        num_mat_out = 0
        num_presence_out = 0
        for node in self.nodes:
            if node.type is NodeType.INPUT_X or node.type is NodeType.INPUT_Y or node.type is NodeType.INPUT_Z or node.type is NodeType.INPUT_D or node.type is NodeType.INPUT_B:
                num_inputs+=1
            elif node.type is NodeType.MATERIAL_OUTPUT:
                num_mat_out+=1
            elif node.type is NodeType.PRESENCE_OUTPUT:
                num_presence_out+=1
        
        if num_inputs != 5 or num_mat_out != 1 or num_presence_out != 1 or (num_presence_out + num_mat_out != 2):
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
    
    #for i in range(8*8*7):
        #a.run(i)
        #print(f"Material: {a.material}")
        #print(f"Presence: {a.presence}")
        #print("--------------")
    
    b = a.to_phenotype()

    newarr = b.reshape(8,8,7)

    print(newarr)
    print(a.valid())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    data = newarr
    z,x,y = data.nonzero()

    ax.scatter(x, y, z, c=z, alpha=1)
    plt.show()
