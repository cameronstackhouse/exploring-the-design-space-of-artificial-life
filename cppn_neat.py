"""
CPPN-NEAT implemented using the neat-python package
"""
#%%

import sys
import os
import pickle
import neat
from typing import List
from tools.evaluate import Run
from tools.activation_functions import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs

def run(
    config_file: str, 
    run_name: str, 
    generations: int = 100,
    size_params: List = [8,8,7], 
    ) -> None:
    """
    Function to run CPPN-NEAT to create and evolve xenobots
    
    :param config_file: Name of config file with CPPN-NEAT hyperparameters
    :param run_name: Name of the run to store xenobot files under
    :param size_params: Size parameters of xenobots being designed
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
    winner = population.run(run.evaluate, generations)
    
    median = stats.get_fitness_median()
    mean = stats.get_fitness_mean()
    std_dev = stats.get_fitness_stdev()
    best_each_gen = stats.get_fitness_stat(fittest_in_gen)
    
    results = {
        "winner": winner,
        "best_each_gen": best_each_gen,
        "mean": mean,
        "median": median,
        "std_dev": std_dev
    }
    
    with open(f"{run_name}.pickle", "wb") as f:
        pickle.dump(results, f)

def fittest_in_gen(scores):
    """ 
    Gets the fittest in the generation
    """
    return max(scores)
    
if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
    