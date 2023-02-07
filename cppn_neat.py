"""
Module implementing CPPN-NEAT for the evolution of CPPNs to produce
xenobots
"""

import sys
import os 
from random import choice, random, uniform
from copy import deepcopy
from typing import List
from networks import Node, NodeType, CPPN
from tools.evaluate import evaluate_pop
from tools.speciate import speciate
from tools.draw_cppn import draw_cppn

def initial_mutations(population: List):
    """
    Function to perform initial mutations on a population of xenobots
    """
    pass

def generate_population(
    size_params: List[int], 
    individuals: int
    ) -> List[CPPN]:
    """
    Function to generate a population of basic CPPNs for the production
    of xenobots

    :param size_params: List of size parameters describing the xenobot design space (e.g [8,8,7] is 8x8x7)
    :param individuals: Number of individuals in each generation
    :rtype: List of CPPNs
    :return: Initial population of CPPNs, each with 6 input nodes and 2 output nodes and are fully connected
    """
    # Generates a population of CPPNs
    return [CPPN(size_params) for _ in range(individuals)]

def crossover(
    cppn_a: CPPN,
    cppn_b: CPPN
    ) -> CPPN:
    """
    Function to crossover two given CPPNs as described by
    the NEAT paper proposed by Stanley

    :param cppn_a: first CPPN
    :param cppn_b: second CPPN
    :rtype: CPPN
    :return: Child of the two crossed over CPPNs
    """
    child = None
    less_fit = None

    rounded_a = round(cppn_a.fitness, 2)
    rounded_b = round(cppn_b.fitness, 2)

    # Gets excess and disjoint genes from the fittest parent
    if rounded_a == rounded_b:
        #TODO finish this
        child = deepcopy(cppn_a)
        less_fit = cppn_b
        excess_and_disjoint = []

        for connection in cppn_a.connections:
            found = False
            for connection_b in cppn_b.connections:
                if connection.historical_marking == connection_b.historical_marking:
                    found = True
                    break
            
            if not found:
                excess_and_disjoint.append(connection_b)

        # Randomly take excess and disjoint genes from parents if equal fitness
        for connection in cppn_a.connections:
            for connection_b in cppn_b.connections:
                pass

    elif rounded_a > rounded_b:
        child = deepcopy(cppn_a)
        less_fit = cppn_b
    elif rounded_b > rounded_a:
        child = deepcopy(cppn_b)
        less_fit = cppn_a
    
    # Swaps matching connections weights with a probability 0.5
    for connection in child.connections:
        for connection_b in less_fit.connections:
            if connection.historical_marking == connection_b.historical_marking:
                if random() > 0.5:
                    connection.weight = connection_b.weight
                
                if not connection.enabled or not connection_b.enabled:
                    if random() < 0.7:
                        connection.enabled = False
                        # checks for validity of network
                        #TODO This might not actually catch error, think again
                        try:
                            child.run(0)
                            if child.material is None or child.presence is None:
                                connection.enabled = True
                        except TypeError:
                            #TODO Update this
                            connection.enabled = True

                        child.reset()
                    else:
                        connection.enabled = True
    
    return child

def crossover_population(population: List[CPPN]) -> List[CPPN]:
    """
    Function to perform crossover on a population of CPPNs

    :param population: population of CPPNs
    :rtype: List of CPPNs
    :return: crossed over population
    """
    # TODO Add comments and tidy
    #TODO Ensure this works too
    crossed_over = []

    # 1) Split population into species
    species = speciate(population, 1.0)

    # 2) Eliminate lowest performing members of population 

    # 3) Assign each species different number of offspring
    # In proportion to sum of adjusted fitness of member organisms
    species_fitness = []
    for spec in species:
        total = 0
        for indv in spec:
            total += indv.fitness
        
        species_fitness.append(total)

    fitness_sum = 0
    for spec in species:
        for indv in spec:
            fitness_sum += indv.fitness

    for i in range(len(species_fitness)):
        species_fitness[i] = species_fitness[i] / fitness_sum
    
    # 4) Crossover
    # TODO Get list of models to crossover, CHECK IF THIS WORKS
    models = []
    deletion_num = (len(population) * 0.7)

    for n, spec in enumerate(species_fitness):
        delete_from_species = int(deletion_num * spec)
        current_species = species[n]
        sorted_current = sorted(current_species, key=lambda indv: indv.fitness, reverse=True)

        m = 0
        for individual in current_species:
            if m < delete_from_species:
                models.append(individual)
            m += 1

    # Crossover
    for _ in range(len(population)):
        parent_1 = choice(models)
        parent_2 = choice(models)

        child = crossover(parent_1, parent_2)

        # TODO maybe remove one of the parents

        crossed_over.append(child)

    return crossed_over

def perturb_weight(connection: CPPN.Connection) -> None:
    """
    Function to perturb a weight of a connection

    :param connection: Connection to modify weight of
    """
    connection.weight = connection.weight + (random() * 0.2)
    
def mutate_weight(cppn: CPPN) -> None:
    """
    Function to mutate a weight of a connection

    :param cppn: CPPN to select connection from
    """
    for connection in cppn.connections:
        if random() < 0.1: # If random is less than 0.1
            connection.weight = uniform(-1,1) # Set the weight to a random number between 0 and 1
        else:
            perturb_weight(connection) # Otherwise just perturb the weight

def mutate_activation_function(cppn: CPPN) -> None:
    """
    Function to mutate an activation function of a node

    :param cppn: CPPN to mutate activation function 
    """
    for layer in cppn.nodes[1:-1]: # Iterates through all layers of nodes (apart from input nodes and output nodes)
        for node in layer:
            if random() < 0.1: # Sees if activation threshold mutation rate is met
                #NOTE Could change to "while activation function == old activation function"
                old_activation = cppn.activation_function
                new_activation = old_activation
                
                while old_activation == new_activation:
                    new_activation = choice(cppn.activation_function)
                
                node.activation_function = new_activation # Assigns the node with a random activation function

def add_node(cppn: CPPN) -> None:
    """
    Function to add a node to a CPPN

    :param cppn: CPPN to add a node to 
    """

    #TODO: CREATING CONNECTIONS BETWEEN SAME LAYER!!!!

    # Choose an enabled connection
    enabled = []
    for connection in cppn.connections:
        if connection.enabled:
            enabled.append(connection)
    
    split = choice(enabled)

    # Split connection in half
    activation_func = choice(cppn.activation_functions)
    output_node = split.out
    input_node = split.input

    if (output_node.layer + 1) == input_node.layer:
        # Increment layer numbers of all higher nodes
        for layer in cppn.nodes[input_node.layer:]:
            for node in layer:
                node.layer += 1
        
        # Insert new layer
        cppn.nodes.insert(output_node.layer + 1, [])
    
    new_node = Node(activation_func, NodeType.HIDDEN, cppn, output_node.layer + 1)
    
    # Create 2 new connections
    cppn.create_connection(output_node, new_node, 1)
    cppn.create_connection(new_node, input_node, uniform(-1,1))

    # Disable old connection
    split.enabled = False

def add_connection(cppn: CPPN) -> None:
    """
    Function to add connection to a CPPN

    :param cppn: CPPN to add a new connection to
    """
#TODO ERROR WITH THIS!!
    existing_conn_pairs = []
    for connection in cppn.connections:
        existing_conn_pairs.append((connection.out, connection.input))
    
    valid = False
    trys = 0
    while not valid and trys < 999:
        valid = True
        out_layer = choice(cppn.nodes[:-1])
        out_layer_index = cppn.nodes.index(out_layer)
        in_layer = choice(cppn.nodes[out_layer_index+1:])

        node1 = choice(out_layer)
        node2 = choice(in_layer)

        if (node1, node2) in existing_conn_pairs:
            trys += 1
            valid = False

        if node1.type is NodeType.HIDDEN or node2.type in [NodeType.INPUT_X, NodeType.INPUT_Y, NodeType.INPUT_Z, NodeType.INPUT_D, NodeType.INPUT_B] or node1 is node2:
            trys += 1
            valid = False

        if valid:
            cppn.create_connection(node1, node2, uniform(-1,1)) #NOTE:FLAG

def mutate_population(population: List[CPPN]) -> None:
    """
    Function to mutate a population of CPPNs

    :param population: Population of CPPNs
    """
    # Rates of different forms of CPPN mutation
    new_connection = 0.5
    add_node_rate = 0.2
    mutate_weight_rate = 0.8
    mutate_activation_function_rate = 0.3
    
    # Iterates through population applying mutations
    for cppn in population:

        # Add connection mutation
        if random() < new_connection:
            add_connection(cppn)
        
        # Add node mutation
        if random() < add_node_rate:
            add_node(cppn)
        
        # Mutate weight mutation
        if random() < mutate_weight_rate:
            mutate_weight(cppn)
        
        # Mutate activation function mutation
        if random() < mutate_activation_function_rate:
            mutate_activation_function(cppn)

def evolve(file_name: str):
    """
    Function to perform CPPN-NEAT to evolve a population of xenobots
    """
    population = generate_population([8,8,7], 50) # Generates an initial population of 

    os.system(f"rm -f fitnessFiles/{file_name}/evaluation.log") # Removes log file if exists
    os.system(f"rm -rf fitnessFiles/{file_name}/bestSoFar")
    os.system(f"mkdir fitnessFiles/{file_name}/bestSoFar")
    generations_complete = 0
    max_fitness = 0

    while generations_complete < 100:
        pass
        # Evaluate population (use evaluate.py)
        evaluate_pop(population, file_name, generations_complete, "fitness_function", max_fitness)

        for indv in population:
            if indv.fitness > max_fitness:
                max_fitness = indv.fitness

        # Speciate 
        speciate(population, 3.4)
        
        # Mutate and crossover fittest
        mutate_population(population)

        # Crossover the population of CPPNs
        population = crossover_population(population)

        # Repeat
        generations_complete += 1

def run_experiment(filename):
    """
    
    """
    evolve(filename)

if __name__ == "__main__":
    #run_experiment(sys.argv[1])
    cppn = CPPN([8,8,7])

    for _ in range(10):
        add_node(cppn)
    
    # for _ in range(100):
    #     add_connection(cppn)
    

