"""
Module to simulate CPPN-NEAT evolution on a population of
CPPNs
"""

from random import uniform, choice
from networks import CPPN, NodeType, Node

#TODO ADD LINE BACK IN
#from tools.evaluate import evaluate_pop

#TODO Evolve CPPNs using modified NEAT

def evolve(population_size, add_node_rate, mutate_node_rate, remove_node_rate, add_edge_rate, mutate_edge_rate, 
    remove_edge_rate, truncation_rate, generations, run_directory, size_params):
    #TODO Write description
    """
    
    """
    population = [] #List containing the population
    fittest = None #Fittest individual

    for _ in range(population_size):
        population.append(CPPN(size_params)) #Sets the initial population
    
    generations_complete = 0 #Counter of number of completed generations
    while generations_complete < generations:
        population = crossover_pop(population) #Crosses over the population
        population = mutate_population(population, add_node_rate, mutate_node_rate, remove_node_rate, add_edge_rate, mutate_edge_rate, remove_edge_rate) #Mutates the population
       
        #TODO Add speciation (Different groups of populations, use historical markings)

        #Evaluates the population using voxcraft-sim to find fitness of each solution
        #TODO ADD LINE BELLOW BACK IN
        #evaluate_pop(population, run_directory, generations_complete, truncation_rate)
                
        #Checks to see if a new overall fittest individual was produced
        for individual in population:
            if fittest == None or individual.fitness > fittest.fitness:
                fittest = individual
        
        generations_complete+=1 #Increments generations counter
    
    return population, fittest #Returns the fittest individual and the population

def crossover_indv(cppn_a: CPPN, cppn_b: CPPN) -> None:
    """
    Function to crossover weights of two CPPNs.
    Only weights with the same innovation number are crossed over
    using a Pseudo uniform crossover.

    :param cppn_a: first CPPN to be crossed over
    :param cppn_b: second CPPN to be crossed over
    """
    #Compares each weight in each connection in each network and crosses over the weights if the innovation numbers match

    for connection_a in cppn_a.connections: #Iterates through all connections in the first cppn
        historical_marking = connection_a.historical_marking #Gets the historical marking of the connection
        for connection_b in cppn_b.connections: #Iterates through all connections in the second cppn
            if historical_marking == connection_b.historical_marking: #Compares the two connections historical markings
                #If the markings match then the weights are swapped
                temp = connection_a.weight
                connection_a.weight = connection_b.weight
                connection_b.weight = temp
                break

def crossover_pop(population, population_size):
    """
    
    """
    #TODO Add description
    return [crossover_indv(choice(population), choice(population)) for _ in range(population_size)]

def mutate_node(node):
    """
    
    """
    #TODO Add description
    #Only mutates non output node activation functions as output node activation functions are always sigmoid functions
    if node.type != NodeType.MATERIAL_OUTPUT or node.type != NodeType.PRESENCE_OUTPUT:
        node.activation_function = choice(node.outer_cppn.activation_functions)

def mutate_nodes(population, rate):
    """
    
    """
    #TODO Add description and comments
    for cppn in population:
        for node in cppn.nodes:
            if rate >= uniform(0,1):
                mutate_node(node)

def add_node_rand_connection(cppn):
    """
    
    """
    #TODO Add comments
    function = choice(cppn.activation_functions)
    cppn.add_node(Node(function, NodeType.HIDDEN, cppn))
    valid = False
    while not valid:
        out = choice(cppn.nodes)
        if out.type != NodeType.MATERIAL_OUTPUT and out.type != NodeType.PRESENCE_OUTPUT:
            cppn.create_connection(out, cppn.nodes[len(cppn.nodes) - 1], uniform(0,1))

def add_node_pop(population, rate):
    """
    
    """
    #TODO Add comments
    for cppn in population:
        if rate >= uniform(0,1):
            add_node_rand_connection(cppn)

def mutate_connection(connection):
    """
    
    """
    #TODO Add description
    connection.weight = uniform(0,1)

def mutate_connections(population, rate):
    """

    """
    #TODO Add description
    for cppn in population:
        for connection in cppn.connections:
            if rate >= uniform(0,1):
                mutate_connection(connection)

def remove_connection(connection):
    #TODO 
    #TODO Check for validity of removal
    pass

def remove_connections(population, rate):
    #TODO Add comments
    for cppn in population:
        for connection in cppn:
            if rate >= uniform(0,1):
                remove_connection(connection)

def add_connection(cppn):
    #TODO
    #TODO Check for validity, cycles
    cppn.prune()
    pass

def add_connections(population, rate):
    #TODO Add comments
    for cppn in population:
        if rate >= uniform(0,1):
            add_connection(cppn)

def remove_nodes(population, rate):
    #TODO Add comments
    #TODO Make functionality work in networks.py
    for cppn in population:
        if rate >= uniform(0,1):
            to_remove = choice(cppn.nodes)
            cppn.remove_node(to_remove)

def mutate_population(population,  add_node_rate, mutate_node_rate, remove_node_rate, add_edge_rate, mutate_edge_rate, remove_edge_rate):
    #TODO Complete functions to make work

    add_node_pop(population, add_node_rate) #Adds nodes to each cppn
    remove_nodes(population, remove_node_rate) #Removes nodes from cppns
    mutate_nodes(population, mutate_node_rate) #Mutates nodes in each cppn
    add_connection(population, add_edge_rate) #Adds edges to cppns
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
    population = []
    for _ in range(100):
        population.append(CPPN([8,8,7]))
    
    population = crossover_pop(population, 100)