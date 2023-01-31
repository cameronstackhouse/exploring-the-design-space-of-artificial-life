"""
Module to simulate CPPN-NEAT evolution on a population of
CPPNs
"""
#TODO maticulously go through this, ensure everything works as intended
#TODO 1) Change mutation probabilities to mutation numbers
#TODO 2) Update docstrings after 1

from copy import copy, deepcopy
from random import randint, uniform, choice, random, gauss
from typing import List
from networks import CPPN, NodeType, Node
from matplotlib import pyplot as plt
from tools.draw_cppn import draw_cppn
from tools.read_files import read_settings
from tools.evaluate import evaluate_pop
from tools.speciate import speciate 
from tools.fitness_functions import FitnessFunction

def evolve(
    population_size: int, 
    add_node_rate: int, 
    mutate_node_rate: int, 
    remove_node_rate: int, 
    add_edge_rate: int, 
    mutate_edge_rate: int, 
    remove_edge_rate: int, 
    truncation_rate: float, 
    generations: int, 
    run_directory: str, 
    size_params: list, 
    fitness_function: FitnessFunction
    ):
    """
    Function to evolve a population of CPPNs using CPPN-NEAT to design xenobots.

    :param population_size: Size of population of CPPNs
    :param add_node_rate: Rate at which a node is added to a CPPN
    :param mutate_node_rate: Rate at which a node is mutated in a CPPN
    :param remove_node_rate: Rate at which a node is removed from a CPPN
    :param add_edge_rate: Rate at which an edge is added to a CPPN
    :param mutate_edge_rate: Rate at which an edge is mutated in a CPPN
    :param remove_edge_rate: Rate at which an edge is removed from a CPPN
    :param truncation_rate: Rate at which the population is truncated (rate at which the population converges on its fittest designs)
    :param generations: Number of generations to evolve the population of CPPNs
    :param run_directory: Directory to save generated history files from voxcraft-sim to
    :param size_params: Size dimentions of the xenobot being created
    :param fitness_function: Fitness function to evaluate each individual in the population
    :rtype: Tuple -> (List of CPPNs, CPPN)
    :return: Tuple containing a list of all CPPNs in the population at the end of evolution and the individual with the highest fitness score
    """
    fittest = None #Fittest individual
    population = create_population(population_size, size_params) #Generates an initial population of CPPNs

    generations_complete = 0 #Counter of number of completed generations
    evaluate_pop(population, run_directory, "Initial", truncation_rate)  #Calculates the initial fitness of the population
    
    while generations_complete < generations: #Repeats until the number of generations specified is reached
        population = select_population(population, population_size, truncation_rate) #Selects the top fit trunction_rate% of the population
        
        print(generations_complete) #Displays how many generations have been completed

        population = crossover_pop(population, population_size) #Crosses over the population
        mutate_population(population, add_node_rate, mutate_node_rate, remove_node_rate, add_edge_rate, mutate_edge_rate, remove_edge_rate) #Mutates the population
       
        print(len(population))
        #Evaluates the population using voxcraft-sim to find fitness of each solution 
        evaluate_pop(population, run_directory, generations_complete, fitness_function)

        #Speciates the population, adjusting fitness of individuals so that topological innovation is protected
        #TODO Check if speciating in right place
        speciate(population, 2)

        #Checks to see if a new overall fittest individual was produced
        for individual in population:
            if fittest == None or individual.fitness > fittest.fitness:
                fittest = individual
        
        generations_complete+=1 #Increments generations counter
    
    #TODO Add pruning maybe
    
    return population, fittest #Returns the fittest individual and the population

def create_population(
    population_size: int, 
    size_params: list
    ) -> None:
    """
    Function to create a population of CPPNs.

    :param population_size: number of CPPNs in the population
    :param size_params: list representing the dimentions of the design space
    """
    population = [CPPN(size_params) for _ in range(population_size)] #Generates a list containing the population of CPPNs
    initial_mutations(population) #Performs initial mutations on the population of cppns
    return population

def select_population(
    population: list, 
    population_size: int, 
    truncation_rate: float
    ) -> List[CPPN]:
    """
    Function to select the suitably fit individuals in the population.

    :param population: population of CPPNs
    :param population_size: size of population to generate
    :param truncation_rate: rate at which the population is truncated (higher rate means fewer, fitter, individuals are chosen)
    :rtype: List 
    :return: List containing the top fittest individuals in the population
    """
    sorted_pop = sorted(population, key=lambda indv: indv.fitness) #Sorts the population by their phenotypes respective fitness scores
    return sorted_pop[:int(population_size*truncation_rate)] #Gets the top fittest individuals of the population and adds them to a list

def initial_mutations(pop):
    """
    
    """

    for cppn in pop:
    
        for _ in range(10):
            #Node adds
            add_node_between_con(cppn)
            
        for _ in range(10):
            #Link adds
            add_connection(cppn)
    
        for _ in range(5):
            #Link removals
            connection = choice(cppn.connections)
            remove_connection(cppn, connection)
    
        for _ in range(100):
            #Node mutations
            layer = choice(cppn.nodes[1:])
            node = choice(layer)
            mutate_node(node)
    
        for _ in range(100):
            #Weight changes
            connection = choice(cppn.connections)
            mutate_connection(connection)


def crossover_indv(
    cppn_a: CPPN, 
    cppn_b: CPPN
    ) -> CPPN:
    """
    Function to crossover connections of two CPPNs.
    Only connections with the same innovation number are crossed over
    using a Pseudo uniform crossover. The excess and disjoint connections
    are maintained from the fitter CPPN.

    :param cppn_a: first CPPN to be crossed over
    :param cppn_b: second CPPN to be crossed over
    :rtype: CPPN
    :return: Crossed over CPPN
    """
    #TODO GENES ARE RANDOMLY CHOSEN AT MATCHING GENES, EXCESS AND DISJOINT
    #TODO I think this is what is going wrong, deepcopy (or copy) for cppn
    # TAKEN FROM MORE FIT PARENT
    #TODO CLEAN CODE!

    child = None

    if cppn_a.fitness > cppn_b.fitness: # Case where the fitness of a is greater than b
        # Takes the excess and disjoint genes from cppn_a
        child = deepcopy(cppn_a)

        # Inherits matching connections with a probability 0.5 from either parent
        for connection in child.connections:
            for connection_b in cppn_b.connections:
                if connection.historical_marking == connection_b.historical_marking:
                    if random() >= 0.5:
                        connection.weight = connection_b.weight
        
    elif cppn_b.fitness > cppn_a.fitness: # Case where the fitness of b is greater than a
        # Takes the excess and disjoint genes from cppn_b
        child = deepcopy(cppn_b)
        
        # Inherits matching connections with a probability 0.5 from either parent
        for connection in child.connections:
            for connection_b in cppn_a.connections:
                if connection.historical_marking == connection_b.historical_marking:
                    if random() >= 0.5:
                        connection.weight = connection_b.weight
    else:
        # Takes the excess and disjoint genes randomly
        if 0.5>= random():
            child = deepcopy(cppn_a)
            for connection in child.connections:
                for connection_b in cppn_b.connections:
                    if connection.historical_marking == connection_b.historical_marking:
                        if random() >= 0.5:
                            connection.weight = connection_b.weight
        else:
            child = deepcopy(cppn_b)
            for connection in child.connections:
                for connection_b in cppn_a.connections:
                    if connection.historical_marking == connection_b.historical_marking:
                        if random() >= 0.5:
                            connection.weight = connection_b.weight
    
    return child

def crossover_pop(
    population: List, 
    population_size: int) -> List[CPPN]:
    """
    Function to crossover the connections in CPPNs in
    a population.

    :param population: population of CPPNsx
    :param population_size: size of population to generate
    :rtype: List of CPPNs
    :return: List of CPPNs with weights crossed over
    """
    return [crossover_indv(choice(population), choice(population)) for _ in range(population_size)]

def mutate_node(node: Node) -> None:
    """
    Function to mutate a node in a CPPN by changing
    its activation function

    :param node: node to mutate
    """
    #Only mutates non output node activation functions as output node activation functions are always sigmoid functions
    old_activation = node.activation_function
    activation_function = node.activation_function    
    
    while activation_function is old_activation:
        activation_function = choice(node.outer.activation_functions)
    
    node.activation_function = activation_function #Sets its activation function as a random activation function

def mutate_nodes(
    population: List, 
    num_mutations: int) -> None:
    """
    Function to mutate nodes within a population.

    :param population: population of CPPNs
    :param rate: rate at which a node is mutated
    """
    for _ in range(num_mutations):
        cppn = choice(population)
        layer = choice(cppn.nodes[1:-1])
        node = choice(layer)
        mutate_node(node)

def add_node_between_con(cppn: CPPN) -> None:
    """
    Function to add a node in the middle of an enabled 
    connection in a CPPN.

    :param cppn: CPPN to add the node
    """
    new_node = None

    #Finds all connections that are enabled
    enabled_connections = []
    for connection in cppn.connections:
        if connection.enabled:
            enabled_connections.append(connection)

    connection = choice(enabled_connections) #Chooses a random connection from enabled connections
    
    #Gets the input and output nodes of the connection
    out = connection.out
    input = connection.input

    weight = connection.weight

    #Finds the index of the layer which the output node is in
    counter = 0
    for i in range(len(cppn.nodes)):
        if out in cppn.nodes[i]:
            counter = i
            break

    out_layer = counter

    #Checks if a new layer needs to be inserted
    if out_layer+1 == len(cppn.nodes)-1: #Checks if the next layer is the output layer
        cppn.nodes.insert(out_layer+1, []) #Inserts an empty layer next to the output layer
    elif input in cppn.nodes[out_layer+1]: #Checks if next layer contains the input node
        cppn.nodes.insert(out_layer+1, []) #If so inserts an empty layer at that position
 
    new_node = Node(choice(cppn.activation_functions), NodeType.HIDDEN, cppn, out_layer+1) #Creates a new node and adds it to the correct layer
    new_node.previous_in = connection.input
    new_node.previous_out = connection.out

    connection_out = connection.out
    connection_input = connection.input

    connection.set_enabled(False) #Disables the old connection

    #Creates two new connection, emulating the node being placed in the middle of the old connection
    cppn.create_connection(connection_out, new_node, 1) 
    cppn.create_connection(new_node, connection_input, weight)

def add_node_pop(
    population: List, 
    num_of_additions: int
    ) -> None:
    """
    Function to add nodes to a population

    :param population: population of CPPNs
    :param rate: rate at which nodes are added to a CPPN
    """
    for _ in range(num_of_additions):
        cppn = choice(population)
        add_node_between_con(cppn)

def mutate_connection(connection: CPPN.Connection) -> None:
    """
    Function to mutate a connection in a CPPN

    :param connection: connection to mutate
    """
    old_weight = connection.weight
    new = old_weight
    while new == old_weight:
        new = gauss(old_weight, 0.5)
        new = max(-1.0, min(new, 1.0))
    connection.weight = new

def mutate_connections(
    population: List, 
    num_of_mutations: int
    ) -> None:
    """
    Function to mutate connections in a population

    :param population: population of CPPNs
    :param rate: rate of connection mutation
    """
    for _ in range(num_of_mutations):
        cppn = choice(population)
        connection = choice(cppn.connections)
        mutate_connection(connection)

def remove_connection(
    cppn: CPPN, 
    connection: CPPN.Connection
    ) -> None:
    """
    Function to remove a given connection from a CPPN

    :param cppn: CPPN to remove connection from
    :param connection: Connection to be removed
    """

    #TODO CHANGE THIS TO USE DFS TO CHECK!! MUCH FASTER. 

    if connection.enabled: #Checks that the connection is enabled
        valid = True
        for node in cppn.nodes[-1]: #Iterates through all output nodes in the CPPN
            connection_counter = 0
            #Checks if the connection being deleted is one that goes into an output node and, if so, checks if it is the only connection going into it
            for connection_check in cppn.connections:
                if connection_check.input is node and connection_check.input is connection.input:
                    connection_counter+=1
            
            #If the connection being deleted is going into an output node and is the only connection then the deletion is invalid
            if connection_counter == 1:
                valid = False
                break
        
        #Counters for the number of disabled connections going into the output nodes from input nodes
        input_output_connection_counter_material = 0 
        input_output_connection_counter_presence = 0

        valid_io_counter = False
        for connection in cppn.connections: #Iterates through connections in the CPPN
            #Checks if the connection is going into the presence output node and is disabled and is coming from an input node
            if connection.input.type is NodeType.PRESENCE_OUTPUT and connection.out.type in [NodeType.INPUT_X, NodeType.INPUT_Y, NodeType.INPUT_Z, NodeType.INPUT_B, NodeType.INPUT_D] and not connection.enabled:
                input_output_connection_counter_presence += 1
            #Checks if the connection is going into the material output node and is disabled and is coming from an input node
            elif connection.input.type is NodeType.MATERIAL_OUTPUT and connection.out.type in [NodeType.INPUT_X, NodeType.INPUT_Y, NodeType.INPUT_Z, NodeType.INPUT_B, NodeType.INPUT_D] and not connection.enabled:
                input_output_connection_counter_material += 1
        
        #Ensures that there is always one connection (enabled or disabled) that links directly from an input node to an output node.
        #This is done to ensure that in the event of node or connection deletions, there is always a way to reconnect a pathway from the 
        #input nodes to the output nodes, maintaining a valid topology
        if not (input_output_connection_counter_material <= 1 and connection.input.type is NodeType.MATERIAL_OUTPUT) and not (input_output_connection_counter_presence <= 1 and connection.input.type is NodeType.PRESENCE_OUTPUT):
            valid_io_counter = True

        #TODO MAYBE CHANGE THIS!
        
        if valid and valid_io_counter: #Checks if the connection is deemed to be valid to be removed
            cppn.connections.remove(connection) #If so the connection can be removed
        try:
            cppn.run(0) #Tries to run the cppn to ensure that the topology is valid (there is at least one pathway to each of the output nodes)
            if cppn.material == 0.5 or cppn.presence == 0.5 or cppn.material is None or cppn.presence is None:
                cppn.connections.append(connection) #If fails, then re-add the connection as the CPPN is not properly connected and the output nodes are unreachable
        except:
            cppn.connections.append(connection) #Catches if none is produced for either of the outputs

def remove_connections(
    population: List, 
    num_of_removals: int
    ) -> None:
    """
    Function to remove connections from population

    :param population: population of CPPNs
    :param rate: rate at which nodes are removed from a CPPN
    """
    #Iterates through CPPNs in the population
    for _ in range(num_of_removals):
        cppn = choice(population)
        connection = choice(cppn.connections)
        remove_connection(cppn, connection)

def add_connection(cppn: CPPN) -> None:
    """
    Function to add a connection to a CPPN

    :param cppn: CPPN to add connection to
    """
    #NOTE No need to check for cycles due to the layer system employed by the code
    output_layer_index = randint(0,len(cppn.nodes) - 2) #Chooses a random layer to get the output node from
    output = choice(cppn.nodes[output_layer_index]) #Chooses a random node from that layer

    input_layer = cppn.nodes[output_layer_index+1] #Gets the input layer as the layer above the output layer

    input = choice(input_layer) #Chooses an input node from the input layer

    connection_exists = False

    for connection in cppn.connections: #Iterates through connections
        if connection.out is output and connection.input is input: #Checks to ensure a connection between the two nodes does not already exist
            connection_exists = True
            connection.weight = uniform(-1,1)
            break

        if not connection_exists:
            weight = uniform(-1,1) #Get the weight for the new connection
            cppn.create_connection(output, input, weight) #Create the new connection and add it to the CPPN


def add_connections(
    population: List, 
    rate_of_addition: int
    ) -> None:
    """
    Function to add connections to a population of CPPNs

    :param population: population of CPPNs to add connections to
    :param rate: rate at which connections are added to the population of CPPNs
    """
    for _ in range(rate_of_addition):
        cppn = choice(population)
        add_connection(cppn)

def remove_nodes(
    population: List, 
    rate: float
    ) -> None:
    """
    Function to remove nodes from a population of CPPNs

    :param population: population of CPPNs
    :param rate: rate of node removal
    """
    for cppn in population: #Iterates through all CPPNs in the population
        if len(cppn.nodes) > 2: #Checks if there are more than 2 layers in the CPPN as input and output nodes can't be removed
            if rate >= uniform(0,1): #Checks that a random number is less than or equal to the node removal rate
                layer = choice(cppn.nodes[1:-1]) #Chooses a random non-input/output layer to delete a node from
                node = choice(layer) #Chooses a node from the selected layer
                cppn.remove_node(node) #Removes the node
        
                #Removes all connections where the io node(s) do now not exist
                for connection in cppn.connections: #Iterates through all connections
                    out = connection.out
                    input = connection.input
                    out_exists = False
                    in_exists = False
                    #Iterates through layers in the CPPN, checking that the output and input nodes of the connection still exist
                    for layer in cppn.nodes: 
                        if out in layer:
                            out_exists = True
                        elif input in layer:
                            in_exists = True
                    if not (in_exists and out_exists): #Checks if the output and input nodes of a connection still exist
                        cppn.connections.remove(connection) #If not the connection is removed
                
                for connection in cppn.connections: #Iterates through all connections
                    #Checks if the CPPN has a connection that should be renabled after the removal of a node
                    #This is done by re-enabling the connection which the node was previously added to the middle of
                    if connection.out is node.previous_out and connection.input is node.previous_in:
                        connection.set_enabled(True)

                
def mutate_population(
    population: List, 
    add_node_rate: int,
    mutate_node_rate: int, 
    remove_node_rate: int, 
    add_edge_rate: int, 
    mutate_edge_rate: int, 
    remove_edge_rate: int) -> None:
    """
    Function to mutate a population of CPPNs

    :param add_node_rate: rate at which a node is added
    :param mutate_node_rate: rate at which a node is mutated
    :param remove_node_rate: rate at which a node is deleted
    :param add_edge_rate: rate at which an edge is added
    :param mutate_edge_rate: rate at which an edge is mutated
    :param remove_edge_rate: rate at which an edge is removed
    """

    add_node_pop(population, add_node_rate) #Adds nodes to each cppn
    remove_nodes(population, remove_node_rate) #Removes nodes from cppns
    mutate_nodes(population, mutate_node_rate) #Mutates nodes in each cppn
    add_connections(population, add_edge_rate) #Adds edges to cppns
    mutate_connections(population, mutate_edge_rate) #Mutate edges in each cppn
    remove_connections(population, remove_edge_rate) #Removes edges in cppns


if __name__ == "__main__":
    """
    #######################
    TESTING ZONE
    DELETE FOR RELEASE
    #TODO
    #######################
    """

    settings = read_settings("settings")
    evolution_params = settings["evolution_paramaters"]

    pop_size = int(evolution_params["population_size"])
    generations = int(evolution_params["generations"])
    add_node_rate = int(evolution_params["node_add_rate"])
    mutate_node_rate = int(evolution_params["mutate_node_rate"])
    delete_node_rate = int(evolution_params["node_removal_rate"])
    add_connection_rate = int(evolution_params["connection_addition_rate"])
    mutate_connection_rate = int(evolution_params["mutate_connection_rate"])
    remove_connection_rate = int(evolution_params["connection_removal_rate"])
    truncation_rate = float(evolution_params["truncation_rate"])
    size_params = list(evolution_params["size_paramaters"])

 

    a, b = evolve(pop_size, add_node_rate, mutate_node_rate, delete_node_rate,
    add_connection_rate, mutate_connection_rate, remove_connection_rate, truncation_rate,
    generations, "a", size_params, "FITNESS PLACEHOLDER")

    draw_cppn(b, show_weights= True)
 
    print(b.connection_types())

    print("---------------")
    c = b.to_phenotype()
    print("---------------")

    newarr = c.reshape(8,8,7)

    na = b.num_activation_functions()

    for x in na:
        print(f"{x}: {na[x]}")
    
    print(b.num_cells())

    for node in b.nodes[0]:
        print(node.activation_function.__name__)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    data = newarr
    z,x,y = data.nonzero()

    ax.scatter(x, y, z, cmap='coolwarm', alpha=1)
    plt.show()



    
    