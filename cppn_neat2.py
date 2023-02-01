from random import choice, random
from copy import deepcopy
from networks import Node, NodeType, CPPN
from tools.evaluate import evaluate_pop
from tools.speciate import speciate

# TODO Add comments and ensure this works

def initial_mutations(population):
    pass

def generate_population(size_params, individuals):
    return [CPPN(size_params) for _ in range(individuals)]

def crossover(cppn_a, cppn_b):
    child = None
    less_fit = None

    if cppn_a.fitness == cppn_b.fitness:
        pass
    elif cppn_a.fitness >= cppn_b.fitness:
        child = deepcopy(cppn_a)
        less_fit = cppn_b
    elif cppn_b.fitness > cppn_a.fitness:
        child = deepcopy(cppn_b)
        less_fit = cppn_a
    
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

def crossover_population(population):
    # Roulette wheel approach
    probabilities = []
    crossed_over = []

    sum_of_fitness = 0
    for cppn in population:
        sum_of_fitness += cppn.fitness
    
    for cppn in population:
        probabilities.append(cppn.fitness/sum_of_fitness)
    
    for _ in range(len(population)):
        prob_one = random()
        prob_two = random()

        parent_1 = None
        parent_2 = None

        cumulative_prob = 0

        for n in range(len(probabilities)):
            cumulative_prob += probabilities[n]
            if prob_one < cumulative_prob and parent_1 is None:
                parent_1 = population[n]
            
            if prob_two < cumulative_prob and parent_2 is None:
                parent_2 = population[n]
        
        child = crossover(parent_1, parent_2)
        crossed_over.append(child)

    return crossed_over

def perturb_weight(connection):
    connection.weight = connection.weight + (random() * 0.2)
    
def mutate_weight(cppn):
    for connection in cppn.connections:
        if random() < 0.1:
            connection.weight = random()
        else:
            perturb_weight(connection)

def mutate_activation_function(cppn):
    for layer in cppn.nodes[1:]:
        for node in layer:
            if random() < 0.1:
                node.activation_function = choice(cppn.activation_functions)

def add_node(cppn):
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

def add_connection(cppn):
    # 1) Pick two nodes without a connection
    trys = 0
    valid = False

    while trys < 999 and not valid:
        valid = True
        out_layer = choice(cppn.nodes[:-1]) # TODO Fix cannot choose from an empty sequence
        out_index = cppn.nodes.index(out_layer)
        in_layer = choice(cppn.nodes[out_index+1:])

        out = choice(out_layer)
        input = choice(in_layer)

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

def mutate_population(population):
    # NOTE Placeholder values
    new_connection = 0.5
    add_node_rate = 0.2
    mutate_weight_rate = 0.8
    mutate_activation_function_rate = 0.3
    
    for cppn in population:
        if random() < new_connection:
            add_connection(cppn)
        
        if random() < add_node_rate:
            add_node(cppn)
        
        if random() < mutate_weight_rate:
            mutate_weight(cppn)
        
        if random() < mutate_activation_function_rate:
            mutate_activation_function(cppn)

def evolve():
    #TODO
    population = generate_population([8,8,7], 50)
    generations_complete = 0

    while generations_complete < 100:
        pass
        # 1) Evaluate population (use evaluate.py)
        evaluate_pop(population, "a", generations_complete, "fitness_function")

        # 2) Speciate 
        speciate(population, 3)
        
        # 3) Mutate and crossover fittest
        mutate_population(population)

        #TODO Defo a problem with crossover
        population = crossover_population(population)

        # 4) Repeat
        generations_complete += 1


if __name__ == "__main__":
    evolve()
    #TODO FIX NoneType and Float error!! Think this is in crossover!

