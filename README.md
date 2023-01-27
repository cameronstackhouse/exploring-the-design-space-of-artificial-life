# Exploring the Design Space of Artificial Life

## Author
[Cameron Stackhouse](https://github.com/cameronstackhouse)

## Description
This project is exploring the design space of xenobots created using an [evolutionary pipeline proposed by Kriegman et al](https://cdorgs.github.io/). The project aims to discover traits in the genotype-phenotype mapping that helps create a population of viable xenobot designs alongside analysing the effectiveness of compositional pattern producing networks as genetic encodings, and identifying why specific genotypes produce specific types of phenotypic structures.

Alongside exploring the design space of the original pipeline, this project aims to explore the genotype-phenotype relationship when other evolutionary algorithms are used to evolve the population of genotypes, such as HyperNEAT. The code uses [voxcraft-sim](https://github.com/voxcraft/voxcraft-sim) to assess the fitness of xenobots for perfoming given tasks, allowing for the evolution of a population.

Based on: 
* https://github.com/voxcraft/voxcraft-sim 
* https://github.com/skriegman/reconfigurable_organisms
* https://github.com/caitlingrasso/Voxcraft-python

## How to use

### Installation
To use the physics simulation to run voxcraft-sim you MUST have an Nvidia GPU alongside an installation of [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base).

Install all python requirements:
    
    pip install -r requirements

Install voxcraft-sim for xenobot simulations following the instructions for local installation specified on the Github:

    https://github.com/voxcraft/voxcraft-sim

### Settings File

### Running Experiments

### Running Tests
Navigate to the exploring-the-design-space-of-artificial-life file:

    cd exploring-the-design-space-of-artificial-life

and run the following command:

    pytest

This will run the tests found in the tests folder on the source code.

## Further Readings

## License
[MIT License](https://github.com/cameronstackhouse/exploring-the-design-space-of-artificial-life/blob/main/LICENSE)

Copyright (c) 2022 cameronstackhouse
