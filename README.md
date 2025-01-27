# Exploring the Design Space of Artificial Life

## Author
[Cameron Stackhouse](https://github.com/cameronstackhouse)

## Description
This project is exploring the design space of xenobots created using an [evolutionary pipeline proposed by Kriegman et al](https://cdorgs.github.io/).
The project aims to answer the following research questions:
1. What are the most designable phenotypes?
2. Is there bias in the genotype-phenotype map in producing particular outputs?
3. How can we cluster phenotypes and can we identify features of organisms which perform better at particular tasks?
4. Do alternative evolutionary algorithms perfom better in identifying high-performance biological designs?

The code uses [voxcraft-sim](https://github.com/voxcraft/voxcraft-sim) to assess the fitness of xenobots for perfoming given tasks, allowing for the evolution of a population.

Based on: 
* https://github.com/voxcraft/voxcraft-sim 
* https://github.com/skriegman/reconfigurable_organisms
* https://github.com/caitlingrasso/Voxcraft-python
* https://github.com/ukuleleplayer/pureples

## How to use

### Installation
To use the physics simulation to run voxcraft-sim you MUST have an Nvidia GPU alongside an installation of [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base).

Install voxcraft-sim for xenobot simulations following the [instructions for local installation](https://github.com/voxcraft/voxcraft-sim).

After installation of voxcraft-sim place the voxcraft-sim folder into the "exploring-the-design-space-of-artificial-life" directory.

Download: 
* [PUREPLES](https://github.com/ukuleleplayer/pureples) 
* [Voxcraft-python](https://github.com/caitlingrasso/Voxcraft-python)

and place in directory.

Install all python requirements:
    
    pip install -r requirements

### Datasets

The data we used for our clustering (structural motif counts and movement frequency components) is available on [kaggle](https://www.kaggle.com/datasets/cameronstackhouse/xenobots).

### CPPN-NEAT and ES-HyperNEAT
To run CPPN-NEAT using the absolute distance of the center of mass of xenobots as the fitness function run the following command:

    python3 cppn_neat.py config_file_name directory_name

this will save history, output, vxa, and vxd files in the directory "fitnessFiles/directory_name/" as specified and create a .pickle file containing data about the run.

Similarly, to run ES-HyperNEAT run the following command:

    python3 es_hyperneat.py config_file_name directory_name

To specify parameters for each, you can modify the config files (initially named config-xenobots and config-hyperneat respectivley.)

### Genotype-Phenotype Map

Code for generating and analysing genotype-phenotype maps is stored in the module 
gp_map.py. 

To create a genotype-phenotype map, initilise a GenotypePhenotypeMap object.
The methods available for genotype-phenotype map analysis contained in the module gp_map.py are described in the gp_map.html documentation file.
Methods for generating and counting structural motifs, alongside the extraction of xenobot movement path frequency components are contained within this module.

### results.ipynb

results.ipynb contains the results we obtained throughout our research for clustering and CPPN-NEAT vs ES-HyperNEAT performance. 
The .pickle files we use for comparison of ES-HyperNEAT and CPPN-NEAT are contained in "result data" (however your own can be generated by running cppn_neat.py and es_hyperneat.py as described earlier).
The data used for clustering is available on [kaggle](https://www.kaggle.com/datasets/cameronstackhouse/xenobots).

## License
[MIT License](https://github.com/cameronstackhouse/exploring-the-design-space-of-artificial-life/blob/main/LICENSE)

Copyright (c) 2022 cameronstackhouse
