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
        #TODO Add speciation (Different groups of populations, use historical markings)
        for cppn in population:
            #TODO Check for validity after each mutation
            #TODO Check if rand needs to be manually seeded

            #TODO Add node
            if add_node_rate >= uniform(0,1):
                add_node_rand_connection(cppn)

            for node in cppn.nodes:
                #TODO Mutate node 
                if mutate_node_rate >= uniform(0,1):
                    result = mutate_node(node)


                #TODO Remove node
                if remove_node_rate >= uniform(0,1):
                    pass
            
            #TODO Add edge
            if add_edge_rate >= uniform(0,1):
                pass
            
            for connection in cppn.connections:
                #TODO Mutate edge
                if mutate_edge_rate >= uniform(0,1):
                    pass

                #TODO Remove edge
                if remove_edge_rate >= uniform(0,1):
                    pass

            pass
        
        #Prune the network, removing redundant links
        cppn.prune()

        #TODO ADD 2 LINES BELLOW BACK IN
        #population = evaluate_pop(population, run_directory, 
                    #generations_complete, truncation_rate)
        
        #TODO Crossover CPPNS
        
        #Checks to see if a new overall fittest individual was produced
        for individual in population:
            if fittest == None or individual.fitness > fittest.fitness:
                fittest = individual

        generations_complete+=1 #Increments generations counter
    
    return fittest #Returns the fittest individual

def crossover(cppn_a, cppn_b):
    """
    Function to crossover weights of two CPPNs
    Only weights with the same innovation number are crossed over
    Pseudo uniform crossover
    """
    #TODO Add comments
    for connection_a in cppn_a.connections:
        historical_marking = connection_a.historical_marking
        for connection_b in cppn_b.connections:
            if historical_marking == connection_b.historical_marking:
                temp = connection_a.weight
                connection_a.weight = connection_b.weight
                connection_b.weight = temp
                break

def mutate_node(node):
    """
    
    """
    #Only mutates non output node activation functions as output node activation functions are always sigmoid functions
    if node.type != NodeType.MATERIAL_OUTPUT or node.type != NodeType.PRESENCE_OUTPUT:
        node.activation_function = choice(node.outer_cppn.activation_functions)
        return True
    else:
        return False

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

def mutate_connection(connection):
    connection.weight = uniform(0,1)

if __name__ == "__main__":
    pass