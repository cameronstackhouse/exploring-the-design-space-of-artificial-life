"""
Module to simulate CPPN-NEAT evolution on a population of
CPPNs
"""

from random import uniform, choice, randint
from networks import CPPN, NodeType
from tools.evaluate import evaluate_pop

#TODO Evolve CPPNs using modified NEAT

def evolve(population_size, mutation_rate, truncation_rate, generations, run_directory,
            size_params):
    #TODO Write description
    """
    
    
    """
    population = [] #List containing the population
    fittest = None #Fittest individual

    for _ in range(population_size):
        population.append(CPPN(size_params)) #Sets the initial population
    
    generations_complete = 0 #Counter of number of completed generations
    while generations_complete < generations:
        for cppn in population:
            #TODO Check for validity after each mutation
            #TODO Change so different number of each mutation
            
            #TODO Add node
            #TODO Mutate node 
            #TODO Remove node
            #TODO Mutate edge
            #TODO Remove edge
            #TODO Add edge

            pass
        population = evaluate_pop(population, run_directory, 
                    generations_complete, truncation_rate)
        
        #TODO Crossover CPPNS
        
        #Checks to see if a new overall fittest individual was produced
        for individual in population:
            if fittest == None or individual.fitness > fittest.fitness:
                fittest = individual

        generations_complete+=1 #Increments generations counter
    
    return fittest #Returns the fittest individual

def crossover():
    pass

def mutate_node(node):
    #Only mutates non output node activation functions as output node activation functions are always sigmoid functions
    if node.type != NodeType.MATERIAL_OUTPUT or node.type != NodeType.PRESENCE_OUTPUT:
        node.activation_function = choice(node.outer_cppn.activation_functions)

def mutate_connection(connection):
    connection.weight = uniform(0,1)

def disable_connection(connection):
    #TODO
    pass

def prune(cppn):
    #TODO
    pass