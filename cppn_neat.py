"""
CPPN-NEAT implemented using the neat-python package
"""
#%%

import os
import pickle
import neat
import numpy as np
from typing import List
from tools.evaluate import Run
from tools.activation_functions import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs
from tools.fitness_functions import FitnessFunction
from tools.gen_phenotype_from_genotype import genotype_to_phenotype
from visualise_xenobot import show
import matplotlib.pyplot as plt

def run(
    config_file: str, 
    run_name: str, 
    generations: int = 100,
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
    # TODO Set fitness function
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
    winner = population.run(run.evaluate, 250)
    
    median = stats.get_fitness_median()
    mean = stats.get_fitness_mean()
    std_dev = stats.get_fitness_stdev()
    best_each_gen = stats.get_fitness_stat(fitest_in_gen)
    
    results = {
        "winner": winner,
        "best_each_gen": best_each_gen,
        "mean": mean,
        "median": median,
        "std_dev": std_dev
    }
    
    with open("NEAT-250.pickle", "wb") as f:
        pickle.dump(results, f)

def fitest_in_gen(scores):
    """ 
    
    """
    return max(scores)
    
if __name__ == "__main__":
    run("config-xenobots", "run_cppn_neat_new_params")
    
    # config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, "config-xenobots")
    
    #    # Adds additional activation functions to NEAT to be able to implement CPPN-NEAR
    # config.genome_config.add_activation("neg_abs", neg_abs)
    # config.genome_config.add_activation("neg_square", neg_square)
    # config.genome_config.add_activation("sqrt_abs", sqrt_abs)
    # config.genome_config.add_activation("neg_sqrt_abs", neg_sqrt_abs)
    