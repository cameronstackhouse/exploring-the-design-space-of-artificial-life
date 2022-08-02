"""
Module to simulate CPPN-NEAT evolution on a population of
CPPNs
"""

from random import randint, uniform, choice
from typing import List
from networks import CPPN, NodeType, Node
from matplotlib import pyplot as plt
from tools.draw_cppn import draw_cppn
from tools.read_outputs import read_settings

#TODO ADD LINE BACK IN
#from tools.evaluate import evaluate_pop

def evolve(population_size, add_node_rate, mutate_node_rate, remove_node_rate, add_edge_rate, mutate_edge_rate, 
    remove_edge_rate, truncation_rate, generations, run_directory, size_params, fitness_function):
    #TODO Write description
    """
    
    """
    fittest = None #Fittest individual
    population = create_population(population_size, size_params) #Generates an initial population of CPPNs

    #Calculates the initial fitness of the population

    #TODO ADD LINE BELLOW BACK IN
    #evaluate_pop(population, run_directory, generations_complete, truncation_rate)
    
    generations_complete = 0 #Counter of number of completed generations
    while generations_complete < generations:

        #TODO Add back in bellow
        #population = select_population(population, population_size, truncation_rate) #Selects the top fit trunction_rate% of the population

        print(generations_complete)

        crossover_pop(population, population_size) #Crosses over the population
        mutate_population(population, add_node_rate, mutate_node_rate, remove_node_rate, add_edge_rate, mutate_edge_rate, remove_edge_rate) #Mutates the population
       
        #TODO Add speciation (Different groups of populations, use historical markings)
        #TODO Change trunctation of population to fit with speciation
        #species = speciate(population) #TODO Add back in

        #Evaluates the population using voxcraft-sim to find fitness of each solution
        #TODO Add lines back in bellow
        #for population in species:
            #evaluate_pop(population, run_directory, generations_complete, fitness_function)

        #Checks to see if a new overall fittest individual was produced
        # for population in species:
            # for individual in population:
            #     if fittest == None or individual.fitness > fittest.fitness:
            #         fittest = individual
        
        generations_complete+=1 #Increments generations counter
    
    #TODO Prune the population of nodes
    for cppn in population:
        cppn.prune()
    
    print(CPPN.innovation_counter)
    
    #fittest.prune()
    return population, fittest #Returns the fittest individual and the population

def create_population(population_size: int, size_params: List) -> None:
    """
    Function to create a population of CPPNs.

    :param population_size: number of CPPNs in the population
    :param size_params: list representing the dimentions of the design space
    """
    return [CPPN(size_params) for _ in range(population_size)] #Generates a list containing the population of CPPNs

def select_population(population: list, population_size: int, truncation_rate: float) -> List:
    """
    Function to select the suitably fit individuals in the population.

    :param population: population of CPPNs
    :param population_size: size of population to generate
    :param truncation_rate: rate at which the population is truncated (higher rate means fewer, fitter, individuals are chosen)
    :rtype: List 
    :return: List containing the top fittest individuals in the population
    """
    sorted_pop = sorted(population, key=lambda indv: indv.fitness) #Sorts the population by their phenotypes respective fitness scores
    return [individual for individual, _ in sorted_pop[:int(population_size*truncation_rate)]] #Gets the top fittest individuals of the population and adds them to a list

def crossover_indv(cppn_a: CPPN, cppn_b: CPPN) -> None:
    """
    Function to crossover weights of two CPPNs.
    Only weights with the same innovation number are crossed over
    using a Pseudo uniform crossover.

    :param cppn_a: first CPPN to be crossed over
    :param cppn_b: second CPPN to be crossed over
    """
    #Compares each weight in each connection in each network and crosses over the weights if the innovation numbers match

    fittest = None
    #TODO USE THIS!
    # if cppn_a.fitness >= cppn_b.fitness:
    #     fittest = cppn_a
    # elif cppn_b.fitness > cppn_a.fitness:
    #     fittest = cppn_b

    #TODO Change to include disjoint and excess weights of more fit parent
    for connection_a in cppn_a.connections: #Iterates through all connections in the first cppn
        historical_marking = connection_a.historical_marking #Gets the historical marking of the connection
        for connection_b in cppn_b.connections: #Iterates through all connections in the second cppn
            if historical_marking == connection_b.historical_marking: #Compares the two connections historical markings
                #If the markings match and the crossover threshold is met then the weights are swapped
                if uniform(0,1) >= 0.5:
                    temp = connection_a.weight
                    connection_a.weight = connection_b.weight
                    connection_b.weight = temp
                    break

def crossover_pop(population: List, population_size: int) -> None:
    """
    Function to crossover the weights of connections in 
    a population.

    :param population: population of CPPNs
    :param population_size: size of population to generate
    """
    for _ in range(population_size): #Iterates through population size
        crossover_indv(choice(population), choice(population)) #Crosses over two random individuals from the population

def mutate_node(node: Node) -> None:
    """
    Function to mutate a node in a CPPN by changing
    its activation function

    :param node: node to mutate
    """
    #Only mutates non output node activation functions as output node activation functions are always sigmoid functions
    if node.type != NodeType.MATERIAL_OUTPUT and node.type != NodeType.PRESENCE_OUTPUT:
        activation_function = choice(node.outer.activation_functions)
        node.activation_function = activation_function #Sets its activation function as a random activation function

def mutate_nodes(population: List, rate: float) -> None:
    """
    Function to mutate nodes within a population.

    :param population: population of CPPNs
    :param rate: rate at which a node is mutated
    """
    for cppn in population: #Iterates through all CPPNs in a population
        for layer in cppn.nodes: #Iterates through all layers in a cppn
            for node in layer: #Iterates through all nodes in a layer
                if rate >= uniform(0,1): #If a random number is less than or equal to the mutation rate
                    mutate_node(node) #Mutate the current node

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
    cppn.create_connection(connection_out, new_node, uniform(0,1)) 
    cppn.create_connection(new_node, connection_input, uniform(0,1))

def add_node_pop(population: List, rate: float) -> None:
    """
    Function to add nodes to a population

    :param population: population of CPPNs
    :param rate: rate at which nodes are added to a CPPN
    """
    for cppn in population: #Iterates through CPPNs in the population
        if rate >= uniform(0,1): #Checks if a random number is less than or equal to the addition rate
            add_node_between_con(cppn) #Adds a node between a random enabled connection

def mutate_connection(connection: CPPN.Connection) -> None:
    """
    Function to mutate a connection in a CPPN

    :param connection: connection to mutate
    """
    connection.weight = uniform(0,1) #Changes the weight to a random value between 0 and 1

def mutate_connections(population: List, rate: float) -> None:
    """
    Function to mutate connections in a population

    :param population: population of CPPNs
    :param rate: rate of connection mutation
    """
    for cppn in population: #Iterates through the population
        for connection in cppn.connections: #Iterates through connections in a CPPN
            if rate >= uniform(0,1): #Checks if a random number is less than or equal to the mutation rate
                mutate_connection(connection) #Mutates the connection

def remove_connection(cppn: CPPN, connection: CPPN.Connection) -> None:
    """
    Function to remove a given connection from a CPPN

    :param cppn: CPPN to remove connection from
    :param conenction: Connection to be removed
    """
    #TODO Add comments
    #TODO works but not computationally feasable!
    cppn.connections.remove(connection)

    cppn.run(0)
    if cppn.material == 0.5 or cppn.presence == 0.5:
        cppn.connections.append(connection)
  
    cppn.connections.append(connection)

def remove_connections(population: List, rate: float) -> None:
    """
    Function to remove connections from population

    :param population: population of CPPNs
    :param rate: rate at which nodes are removed from a CPPN
    """
    #Iterates through CPPNs in the population
    for cppn in population:
        for connection in cppn.connections:
            if rate >= uniform(0,1): #Checks if a random number is less than or equal to the removal rate
                remove_connection(cppn, connection) #If so the connection is removed from the CPPN

def add_connection(cppn: CPPN) -> None:
    """
    Function to add a connection to a CPPN

    :param cppn: CPPN to add connection to
    """

    output_layer_index = randint(0,len(cppn.nodes) - 2) #Chooses a random layer to get the output node from
    output = choice(cppn.nodes[output_layer_index]) #Chooses a random node from that layer

    input_layer = cppn.nodes[output_layer_index+1] #Gets the input layer as the layer above the output layer

    input = choice(input_layer) #Chooses an input node from the input layer

    connection_exists = False

    for connection in cppn.connections: #Iterates through connections
        if connection.out is output and connection.input is input: #Checks to ensure a connection between the two nodes does not already exist
            connection_exists = True
            connection.weight = uniform(0,1)
            break

        if not connection_exists:
            weight = uniform(0,1) #Get the weight for the new connection
            cppn.create_connection(output, input, weight) #Create the new connection and add it to the CPPN


def add_connections(population: List, rate: float) -> None:
    """
    Function to add connections to a population of CPPNs

    :param population: population of CPPNs to add connections to
    :param rate: rate at which connections are added to the population of CPPNs
    """
    for cppn in population: #Iterates through the population of CPPNs
        if rate >= uniform(0,1): #Checks if the random number is less than or equal to the addition rate
            add_connection(cppn) #Adds a connection to the CPPN

def remove_nodes(population: List, rate: float) -> None:
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

def cppn_distance(cppn1, cppn2):
    #TODO
    pass

def speciate(population) -> List[List]:
    #TODO Create function to speciate a population
    pass
                

def mutate_population(population: List, add_node_rate: float, mutate_node_rate: float, remove_node_rate: float, add_edge_rate: float, 
    mutate_edge_rate: float, remove_edge_rate: float) -> None:
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
    add_node_rate = float(evolution_params["node_add_rate"])
    mutate_node_rate = float(evolution_params["mutate_node_rate"])
    delete_node_rate = float(evolution_params["node_removal_rate"])
    add_connection_rate = float(evolution_params["connection_addition_rate"])
    mutate_connection_rate = float(evolution_params["mutate_connection_rate"])
    remove_connection_rate = float(evolution_params["connection_removal_rate"])
    truncation_rate = float(evolution_params["truncation_rate"])
    size_params = list(evolution_params["size_paramaters"])

    a, b = evolve(pop_size, add_node_rate, mutate_node_rate, delete_node_rate,
    add_connection_rate, mutate_connection_rate, remove_connection_rate, truncation_rate,
    generations, "a", size_params, "FITNESS PLACEHOLDER")

    first = a[8]

    draw_cppn(first, show_weights= True)
 
    print(first.connection_types())

    b = first.to_phenotype()

    newarr = b.reshape(8,8,7)

    na = first.num_activation_functions()

    for x in na:
        print(f"{x}: {na[x]}")
    
    print(first.num_cells())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    data = newarr
    z,x,y = data.nonzero()

    ax.scatter(x, y, z, cmap='coolwarm', alpha=1)
    plt.show()



    
    