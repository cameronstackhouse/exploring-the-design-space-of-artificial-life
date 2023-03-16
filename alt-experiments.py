"""
Script to run and generate output files for all alternative experiments
"""

import es_hyperneat
import cppn_neat
from tools.fitness_functions import FitnessFunction

def run_and_save(
    filename: str = "result data", 
    generations: int = 100,
    cppn_neat_config: str = "config-xenobots",
    es_hyperneat_config: str = "config-hyperneat"
    ) -> None:
    # TODO Before this: Add CPPNs to CSV
    fitness_functions = [FitnessFunction.MAX_DISTANCE, FitnessFunction.OBJECT_EXPULSION, 
                         FitnessFunction.ABS_DISTANCE, FitnessFunction.X_ONLY, FitnessFunction.Y_ONLY,
                         FitnessFunction.WALL_OBSTACLE, FitnessFunction.SMALL_XENOBOTS]
    
    fit_func_names = ["max-distance", "object-expulsion",
                      "abs-distance", "x-only",
                      "y-only", "wall-obsticle", "small-xenobots"]
    
    for n, function in enumerate(fitness_functions):
        #TODO before running ensure fitness functions fully operational
        # CPPN-NEAT 
        cppn_neat.run("config-xenobots", f"run-{fit_func_names[n]}-CPPN-NEAT", generations=generations, fitness_func=function)
        # ES-HyperNEAT 
        es_hyperneat.run("config-hyperneat", f"run-{fit_func_names[n]}-ES-HyperNEAT", generations=generations, fitness_func=function)
        
if __name__ == "__main__":
    run_and_save()