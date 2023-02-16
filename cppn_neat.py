"""
CPPN-NEAT implemented using the neat-python package
"""

import os
import neat
from typing import List
from tools.evaluate import Run
from tools.activation_functions import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs
from tools.fitness_functions import FitnessFunction

def run(
    config_file: str, 
    run_name: str, 
    size_params: List = [8,8,7], 
    fitness_func: FitnessFunction = None
    ) -> None:
    """
    Function to run CPPN-NEAT to create and evolve xenobots
    
    :param config_file: Name of config file with CPPN-NEAT hyperparameters
    :param run_name: Name of the run to store xenobot files under
    :param size_params: Size parameters of xenobots being designed
    :param fitness_func: Fitness function being used for evaluation
    """
    # Creates a config file for NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    # Adds additional activation functions to NEAT to be able to implement CPPN-NEAR
    config.genome_config.add_activation("neg_abs", neg_abs)
    config.genome_config.add_activation("neg_square", neg_square)
    config.genome_config.add_activation("sqrt_abs", sqrt_abs)
    config.genome_config.add_activation("neg_sqrt_abs", neg_sqrt_abs)
    
    # Creates files to store xenobot files and results
    os.system(f"rm -rf fitnessFiles/{run_name}")
    os.system(f"mkdir -p fitnessFiles/{run_name}")
    os.system(f"cp {config_file} fitnessFiles/{run_name}/")
    os.system(f"mv fitnessFiles/{run_name}/{config_file} fitnessFiles/{run_name}/run_parameters")
    
    # Creates a run container passing in run_name and size_params so they can be accessed during evaluation
    run = Run(run_name, size_params=size_params)
    
    # Creates a population for CPPN-NEAT using config file settings
    population = neat.Population(config)
    
    # Adds statistics reporter to CPPN-NEAT, needed for gathering data
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Runs CPPN-NEAT for 
    winner = population.run(run.evaluate, 100)
    
    for node in winner.nodes:
        print(winner.nodes[node].activation)
    

if __name__ == "__main__":
    run("config-xenobots", "run-big")
    
    # Gene object produced by CPPN-NEAT library has nodes and connections self