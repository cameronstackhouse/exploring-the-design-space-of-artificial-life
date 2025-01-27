"""
Module to simulate ES-HyperNEAT on a population of 
CPPNs for the designing of Xenobots.
Based on https://github.com/ukuleleplayer/pureples/tree/master/pureples
"""
#%%
import sys
import os
import pickle
import neat
from typing import List
from pureples.shared.substrate import Substrate
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from tools.evaluate import Run
from tools.activation_functions import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs

def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {"initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
            "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 5.0,
            "activation": "sigmoid"}

def run(
    config_file, 
    run_name, 
    generations = 100,
    version = "L",
    size_params = [8,8,7]
    ) -> None:
    """
    Function to run HyperNEAT to create and evolve xenobots
    
    :param config_file: Configuration file name for NEAT
    :param run_name: Name of the run to store xenobot files
    :param generations: Number of generations 
    :param version: Describes complexity of HyperNEAT run
    :param size_params: Size parameters of xenobots being designed
    """
    INPUT_COORDINATES = []
    
    for i in range(0, 5):
        INPUT_COORDINATES.append((-1 + (2 * i/3), -1))
    
    OUTPUT_COORDINATES = [(-1, 1), (1, 1)]

    SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)
    PARAMS = params(version)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    config.genome_config.add_activation("neg_abs", neg_abs)
    config.genome_config.add_activation("neg_square", neg_square)
    config.genome_config.add_activation("sqrt_abs", sqrt_abs)
    config.genome_config.add_activation("neg_sqrt_abs", neg_sqrt_abs)
    
    os.system(f"rm -rf fitnessFiles/{run_name}")
    os.system(f"mkdir -p fitnessFiles/{run_name}")
    os.system(f"cp {config_file} fitnessFiles/{run_name}/")
    os.system(f"mv fitnessFiles/{run_name}/{config_file} fitnessFiles/{run_name}/run_parameters")
    
    run = Run(run_name, params=PARAMS, substrate=SUBSTRATE, size_params=size_params, hyperneat=True)
    
    population = neat.Population(config)
    
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    winner = population.run(run.evaluate, generations)
    
    CPPN = neat.nn.FeedForwardNetwork.create(winner, config)
    NETWORK = ESNetwork(SUBSTRATE, CPPN, PARAMS)
    
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
    
    with open(f"{run_name}.pickle", "wb") as f:
        pickle.dump(results, f)

def fitest_in_gen(scores: List) -> float:
    """ 
    Gets the fittest individual in a generation
    from a list of fitnesses
    
    :param scores: fitness scores 
    :rtype: list
    :return: greatest fitness score
    """
    return max(scores)

if __name__ == "__main__":
    #run("config-hyperneat", "run_250_hyperneat_FINAL", generations=250)
    run(sys.argv[0], sys.argv[1])
    