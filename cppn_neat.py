"""
CPPN-NEAT implemented using the neat-python package
"""

import os
import neat
from tools.evaluate import Run
from tools.activation_functions import neg_abs, neg_square, sqrt_abs, neg_sqrt_abs

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    config.genome_config.add_activation("neg_abs", neg_abs)
    config.genome_config.add_activation("neg_square", neg_square)
    config.genome_config.add_activation("sqrt_abs", sqrt_abs)
    config.genome_config.add_activation("neg_sqrt_abs", neg_sqrt_abs)
    
    run_name = "test_copy_config"
    
    os.system(f"rm -rf fitnessFiles/{run_name}")
    os.system(f"mkdir -p fitnessFiles/{run_name}")
    os.system(f"cp {config_file} fitnessFiles/{run_name}/")
    os.system(f"mv fitnessFiles/{run_name}/{config_file} fitnessFiles/{run_name}/run_parameters")
    run = Run(run_name)
    
    population = neat.Population(config)
    
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    winner = population.run(run.evaluate, 100)

if __name__ == "__main__":
    run("config-xenobots")