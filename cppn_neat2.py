from random import choice, random
from networks import Node, NodeType
from tools.evaluate import evaluate_pop
from tools.speciate import speciate


def generate_population(size_params, individuals):
    return [CPPN(size_params) for _ in range(individuals)]

def crossover(cppn_a, cppn_b):
    #TODO Crossover function as specified by NEAT paper
    child = None

    if cppn_a.fitness >= cppn_b.fitness:
        pass
    elif cppn_b.fitness > cppn_a.fitness:
        pass
    
    pass

def crossover_population(population):
    pass

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
            if random < 0.1:
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
        out_layer = choice(cppn.nodes[:-1])
        out_index = cppn.nodes.index(out_layer)
        in_layer = choice(cppn.nodes[out_layer+1:])

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
                cppn.pop()
                trys += 1

def mutate_population(population):
    # NOTE Placeholder values
    new_connection = 0.5
    add_node = 0.2
    mutate_weight = 0.8
    mutate_activation_function = 0.3
    
    for cppn in population:
        if random() < new_connection:
            add_connection(cppn)
        
        if random() < add_node:
            add_node(cppn)
        
        if random() < mutate_weight:
            mutate_weight(cppn)
        
        if random() < mutate_activation_function:
            mutate_activation_function(cppn)

def evolve():
    #TODO
    population = generate_population([8,8,7], 100)
    generations_complete = 0

    for while generations_complete < 100:
        pass
        # 1) Evaluate population (use evaluate.py)
        evaluate_pop(population, "a", generations_complete, "fitness_function")

        # 2) Speciate 
        speciate(population, 3)

        # 3) Kill off worst NNs
        # TODO
        
        # 4) Mutate and crossover
        # TODO
        mutate_population(population)
        #population = crossover_population(population)

        # 5) Repeat
        generations_complete += 1


if __name__ == "__main__":
    evolve()

