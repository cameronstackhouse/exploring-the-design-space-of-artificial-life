"""
Module implementing CPPN-NEAT for the evolution of CPPNs to produce
xenobots
"""

from random import choice, random
from copy import deepcopy
from typing import List
from networks import Node, NodeType, CPPN
from tools.evaluate import evaluate_pop
from tools.speciate import speciate


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

    # Gets excess and disjoint genes from the fittest parent
    if cppn_a.fitness == cppn_b.fitness:
        #PLACEHOLDER
        child = deepcopy(cppn_a)
        less_fit = cppn_b
    elif cppn_a.fitness >= cppn_b.fitness:
        child = deepcopy(cppn_a)
        less_fit = cppn_b
    elif cppn_b.fitness > cppn_a.fitness:
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
                        child.run(0)
                        if child.material is None or child.presence is None:
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
    # Roulette wheel selection
    probabilities = []
    crossed_over = []

    sum_of_fitness = 0
    for cppn in population:
        sum_of_fitness += cppn.fitness
    
    for cppn in population:
        probabilities.append(cppn.fitness/sum_of_fitness) # Makes a list containing the fitness of an individual/total fitness to get probability of selection
    
    for _ in range(len(population)):
        prob_one = random()
        prob_two = random()

        parent_1 = None
        parent_2 = None

        cumulative_prob = 0

        # Selects the two parent genotypes using their assigned probability
        for n in range(len(probabilities)):
            cumulative_prob += probabilities[n]
            if prob_one < cumulative_prob and parent_1 is None:
                parent_1 = population[n]
            
            if prob_two < cumulative_prob and parent_2 is None:
                parent_2 = population[n]
        
        child = crossover(parent_1, parent_2) # Crosses over the two parents
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
            connection.weight = random() # Set the weight to a random number between 0 and 1
        else:
            perturb_weight(connection) # Otherwise just perturb the weight

def mutate_activation_function(cppn: CPPN) -> None:
    """
    Function to mutate an activation function of a node

    :param cppn: CPPN to mutate activation function 
    """
    for layer in cppn.nodes[1:]: # Iterates through all layers of nodes (apart from input nodes)
        for node in layer:
            if random() < 0.1: # Sees if activation threshold mutation rate is met
                #NOTE Could change to "while activation function == old activation function"
                node.activation_function = choice(cppn.activation_functions) # Assigns the node with a random activation function

def add_node(cppn: CPPN) -> None:
    """
    Function to add a node to a CPPN

    :param cppn: CPPN to add a node to 
    """
    # 1) Pick a connection 
    if len(cppn.connections) > 0:
        connection = choice(cppn.connections)
        out = connection.out
        input = connection.input
        old_weight = connection.weight

        # 2) Split the connection with a new node with a random activation function
        activation_function = choice(cppn.activation_functions)

        # Adds new layers into structure if necissary
        if out.layer + 1 == len(cppn.nodes) - 1:
            cppn.nodes.insert(out.layer+1, [])

        new = Node(activation_function, NodeType.HIDDEN, cppn, out.layer + 1)

        # 3) Disable the old connection
        connection.enabled = False
        
        # 4) Create 2 new connections, one with weight 1 (into the node) and one with the weight 
        # of the previous connection leaving the node
        cppn.create_connection(out, new, 1)
        cppn.create_connection(new, input, old_weight)

def add_connection(cppn: CPPN) -> None:
    """
    Function to add connection to a CPPN

    :param cppn: CPPN to add a new connection to
    """
    # 1) Pick two nodes without a connection
    trys = 0
    valid = False

    while trys < 999 and not valid:
        valid = True
        out_layer = choice(cppn.nodes[:-1]) 
        out_index = cppn.nodes.index(out_layer)
        in_layer = choice(cppn.nodes[out_index+1:])

        # Picks the two nodes
        out = choice(out_layer)
        input = choice(in_layer)

        # Checks if connection already exists in topology
        for connection in cppn.connections:
            if connection.out is out and connection.input is input:
                trys += 1
                valid = False
                break
        
        # 2) Add connection with random weight
        if valid:
            cppn.create_connection(out, input, random())

        # 3) Ensure a cycle has not been created
            if cppn.has_cycles():
                valid = False
                cppn.connections.pop() 
                trys += 1

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

def evolve():
    """
    Function to perform CPPN-NEAT to evolve a population of xenobots
    """
    population = generate_population([8,8,7], 50) # Generates an initial population of 
    generations_complete = 0

    while generations_complete < 100:
        pass
        # Evaluate population (use evaluate.py)
        evaluate_pop(population, "a", generations_complete, "fitness_function")

        # Speciate 
        speciate(population, 3)
        
        # Mutate and crossover fittest
        mutate_population(population)

        # Crossover the population of CPPNs
        population = crossover_population(population)

        # Repeat
        generations_complete += 1


if __name__ == "__main__":
    evolve()
