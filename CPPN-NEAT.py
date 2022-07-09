"""
Module to simulate CPPN-NEAT evolution on a population of
CPPNs
"""

from random import uniform, choice
from networks import CPPN, NodeType, Node
from matplotlib import pyplot as plt


#TODO ADD LINE BACK IN
#from tools.evaluate import evaluate_pop

#TODO Evolve CPPNs using modified NEAT

def evolve(population_size, add_node_rate, mutate_node_rate, remove_node_rate, add_edge_rate, mutate_edge_rate, 
    remove_edge_rate, truncation_rate, generations, run_directory, size_params):
    #TODO Write description
    """
    
    """
    fittest = None #Fittest individual
    population = create_population(population_size, size_params) #Generates an initial population of CPPNs

    #Calculates the initial fitness of the population
    #TODO ADD LINE BELLOW BACK IN
    #fit_population = evaluate_pop(population, run_directory, generations_complete, truncation_rate)
    
    generations_complete = 0 #Counter of number of completed generations
    while generations_complete < generations:
        #TODO Add back in bellow
        #population = select_population(population, fit_population, population_size, truncation_rate) #Selects the top fit trunction_rate% of the population
        print(generations_complete)
        #TODO Add back in bellow
        #population = select_population(population, fit_population)
        crossover_pop(population, population_size) #Crosses over the population
        mutate_population(population, add_node_rate, mutate_node_rate, remove_node_rate, add_edge_rate, mutate_edge_rate, remove_edge_rate) #Mutates the population
       
        #TODO Add speciation (Different groups of populations, use historical markings)

        #Evaluates the population using voxcraft-sim to find fitness of each solution

        #TODO Add back in bellow
        #fit_population = evaluate_pop(population, run_directory, generations_complete, truncation_rate)
                
        #Checks to see if a new overall fittest individual was produced
        
        # for individual in population:
        #     if fittest == None or individual.fitness > fittest.fitness:
        #         fittest = individual
        
        generations_complete+=1 #Increments generations counter
    
    return population, fittest #Returns the fittest individual and the population

def create_population(population_size, size_params):
    """
    
    """
    #TODO Add comments
    return [CPPN(size_params) for _ in range(population_size)]

def select_population(population, fit_population, population_size, truncation_rate):
    """
    
    """
    #TODO Add comments
    sorted_pop = sorted(zip(population, fit_population), key=lambda fitness: fitness[1])
    return [individual for individual, _ in sorted_pop[:int(population_size*truncation_rate)]]

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
    #TODO Add comments
    for _ in range(population_size):
        crossover_indv(choice(population), choice(population))

def mutate_node(node):
    """
    
    """
    #TODO Add description
    #Only mutates non output node activation functions as output node activation functions are always sigmoid functions
    if node.type != NodeType.MATERIAL_OUTPUT or node.type != NodeType.PRESENCE_OUTPUT:
        node.activation_function = choice(node.outer.activation_functions)

def mutate_nodes(population, rate):
    """
    
    """
    #TODO Add description and comments
    for cppn in population:
        for layer in cppn.nodes:
            for node in layer:
                if rate >= uniform(0,1):
                    mutate_node(node)

def add_node_between_con(cppn):
    """
    
    """
    #TODO Add comments
    #TODO CHANGE TO ADD NODE INBETWEEN A RANDOM CONNECTION AND THEN DISABLE THAT CONNECTION
    #TODO WHEN ADDING A NODE, DFS TO FIND NUMBER OF NODES IN PATH, IF LESS THAN len(nodes) ADD LAYER AND INCREMENT LAYER NUMBER OF ALL NODES ABOVE NEW LAYER
    function = choice(cppn.activation_functions)
    conenction = choice(cppn.connections) 
    

def add_node_pop(population, rate):
    """
    
    """
    #TODO Add comments
    for cppn in population:
        if rate >= uniform(0,1):
            add_node_between_con(cppn)

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

def remove_connection(cppn, connection):
    #TODO 
    #TODO Check for validity of removal. Check at least 1 connection to 2 output nodes
    cppn.connections.remove(connection)
    cppn.prune()
    

def remove_connections(population, rate):
    #TODO Add comments
    for cppn in population:
        for connection in cppn.connections:
            if rate >= uniform(0,1):
                remove_connection(cppn, connection)

def add_connection(cppn):
    #TODO CHANGE TO LAYER SYSTEM
    if len(cppn.nodes) != 7:
        output = choice(cppn.nodes[7:])
        input = choice(cppn.nodes[5:])
        weight = uniform(0,1)
        cppn.create_connection(output, input, weight)
    

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
            #TODO Choose random layer
            to_remove = choice(cppn.nodes)
            cppn.remove_node(to_remove)

def mutate_population(population, add_node_rate, mutate_node_rate, remove_node_rate, add_edge_rate, mutate_edge_rate, remove_edge_rate):
    #TODO Complete functions to make work
    add_node_pop(population, add_node_rate) #Adds nodes to each cppn
    #remove_nodes(population, remove_node_rate) #Removes nodes from cppns #TODO COMPLETE
    mutate_nodes(population, mutate_node_rate) #Mutates nodes in each cppn
    add_connections(population, add_edge_rate) #Adds edges to cppns
    mutate_connections(population, mutate_edge_rate) #Mutate edges in each cppn
    #remove_connections(population, remove_edge_rate) #Removes edges in cppns #TODO COMPLETE       


if __name__ == "__main__":
    """
    #######################
    TESTING ZONE
    DELETE FOR RELEASE
    #TODO
    #######################
    """
    a, b = evolve(100, 0.5, 0.5, 0.1, 0.9, 0.5, 0.1, 0.3, 100, "a", [8,8,7])

    first = a[45]

    for con in first.connections:
        if con.out.type is NodeType.HIDDEN:
            print(con.input.type)

    b = first.to_phenotype()

    newarr = b.reshape(8,8,7)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    data = newarr
    z,x,y = data.nonzero()

    ax.scatter(x, y, z, cmap='coolwarm', alpha=1)
    plt.show()



    
    