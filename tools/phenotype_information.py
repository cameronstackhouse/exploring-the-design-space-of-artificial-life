"""
Module to get information from phenotypes for use in clustering and comparing.
"""

import os
import neat
from scipy.fft import fftn
import numpy as np
from itertools import combinations, permutations
from tools.read_files import read_history
from voxcraftpython.VoxcraftVXA import VXA
from voxcraftpython.VoxcraftVXD import VXD
from tools.activation_functions import normalize
from pureples.es_hyperneat.es_hyperneat import ESNetwork

def KC_LZ(string):
    """
    Lempel-ziv to calculate the complexity of a xenobot phenotype.

    Code taken from https://cclab.science/papers/Nature_Communications_2018.pdf
    """
    n=len(string)
    s = '0'+string
    c=1
    l=1
    i=0
    k=1
    k_max=1
    stop=0

    while stop==0:
        if s[i+k] != s[l+k]:
            if k>k_max:
                k_max=k
            i=i+1
            if i==l:
                c=c+1
                l=l+k_max
                if l+1>n:
                    stop=1
                else:
                    i=0
                    k=1
                    k_max=1
            else:
                k=1
        else:
            k=k+1

            if l+k>n:
                c=c+1
                stop=1

    # a la Lempel and Ziv (IEEE trans inf theory it-22, 75 (1976), 
    # h(n)=c(n)/b(n) where c(n) is the kolmogorov complexity
    # and h(n) is a normalised measure of complexity.
    complexity = c
    return complexity

def calc_KC(s):
    """
    Calculates the complexity of a xenobot phenotype
    """
    L = len(s)
    if s == '0'*L or s == '1'*L or s == '2'*L:
        return np.log2(L)
    else:
        return np.log2(L)*(KC_LZ(s)+KC_LZ(s[::-1]))/2.0

def movement_frequency_components(movement) -> np.ndarray:
    """
    Function to get the frequency components of a xenobot movement path to use in clustering of xenobot behaviour.
    This is done using discrete 3-Dimensional Fourier transform on the X, Y, and Z movement coordinates of the 
    xenobot. 

    :param: CPPN which produces the xenobot
    :return: List of frequency components of the movement path of the xenobot, describing the behaviour of the xenobot
    """
    
    frequency_components = fftn(movement)

    return frequency_components

def motif_vectorisation(phenotype: str) -> list:
    """
    Function to summarize a phenotype by motifs in its structure into a vector to be used in 
    clustering of structure.

    :param phenotype: 
    """
    #TODO Use motifs to vectorize

    motifs = possible_motifs()
    
    return []

def num_cells(phenotype) -> dict:
    """
    Function to return the number of the three different types of cells in the 
    phenotype

    :rtype: dict
    :return dictionary containing number of skin and muscle cells
    """
    none = 0
    skin = 0
    muscle = 0
    #Iterates through phenotype cells and increments the associated cell counter
    for cell in phenotype:
        if cell == 0:
            none+=1
        elif cell == 1:
            skin+=1
        elif cell == 2:
            muscle+=1
        
    return {"none": none, "skin": skin, "muscle": muscle}

def possible_motifs() -> list:
    """
    Generates possible motifs from size 2 to 4
    """
    possible_cells = [0,1,2]
    
    two_cell_motifs = list(combinations(possible_cells, 2))
    three_cell_motifs = []
    four_cell_motifs = []


def movement_components(
    genome, 
    config, 
    size_params, 
    hyperneat = False, 
    substrate = None, 
    hyperneat_params = None
    ):
    """
    Gets the movement components of a xenobot
    from the history file produced by voxcraft-sim
    """
    #TODO change to do multiple cppns. Allow vxa and vxd options
    net = None
    if hyperneat:
        cppn_designer = neat.nn.FeedForwardNetwork.create(genome, config)
        xenobot_producer_network = ESNetwork(substrate, cppn_designer, hyperneat_params)
        net = xenobot_producer_network.create_phenotype_network()
    else:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Make dir to run
    os.system("mkdir temp_run_movement")

    # 1) Make VXD File
    vxa = VXA(SimTime=3, HeapSize=0.65, RecordStepSize=100, DtFrac=0.95, EnableExpansion=1) 

    passive = vxa.add_material(RGBA=(0,255,0), E=5000000, RHO=1000000) # passive soft
    active = vxa.add_material(RGBA=(255,0,0), CTE=0.01, E=5000000, RHO=1000000) # active
    
    vxa.write("base.vxa") #Write a base vxa file
    os.system(f"cp base.vxa temp_run_movement/") #Copy vxa file to correct run directory
    os.system("rm base.vxa") #Removes old vxa file

    # 2) Make VXA File
    x_inputs = np.zeros(size_params)
    y_inputs = np.zeros(size_params)
    z_inputs = np.zeros(size_params)
    
    for x in range(size_params[0]):
            for y in range(size_params[1]):
                for z in range(size_params[2]):
                    x_inputs[x, y, z] = x
                    y_inputs[x, y, z] = y
                    z_inputs[x, y, z] = z

    x_inputs = normalize(x_inputs)
    y_inputs = normalize(y_inputs)
    z_inputs = normalize(z_inputs)

    #Creates the d input array, calculating the distance each point is away from the centre
    d_inputs = normalize(np.power(np.power(x_inputs, 2) + np.power(y_inputs, 2) + np.power(z_inputs, 2), 0.5))

    #Creates the b input array, which is just a numpy array of ones
    b_inputs = np.ones(size_params)

    #Sets all inputs and flattens them into 1D arrays
    x_inputs = x_inputs.flatten()
    y_inputs = y_inputs.flatten()
    z_inputs = z_inputs.flatten()
    d_inputs = d_inputs.flatten()
    b_inputs = b_inputs.flatten()
    
    inputs = list(zip(x_inputs, y_inputs, z_inputs, d_inputs, b_inputs))
    
    for n, input in enumerate(inputs):
            output = net.activate(input)
            presence = output[0]
            material = output[1]
            
            if presence <= 0.2: #Checks if presence output is less than 0.2
                body[n] = 0 #If so there is no material in the location
            elif material < 0.5: #Checks if material output is less than 0.5 
                body[n] = 1 #If so there is skin in the location
            else:
                body[n] = 2 #Else there is a cardiac cell in the location
    
    for cell in range(len(body)):
        if body[cell] == 1:
            body[cell] = passive
        elif body[cell] == 2:
            body[cell] = active 
            
    body = body.reshape(size_params[0], size_params[1], size_params[2])
    vxd = VXD()
    vxd.set_tags(RecordVoxel=1) # pass vxd tags in here to overwite vxa tags
    vxd.set_data(body) #Sets the data to be written as the phenotype generated
    
    vxd.write(f"movement.vxd") #Writes vxd file for individual
    os.system(f"cp movement.vxd temp_run_movement/")
    os.system(f"rm movement.vxd") #Removes the old non-copied vxd file

    # 3) Run voxcraft-sim
    os.chdir("voxcraft-sim/build")
    os.system("./voxcraft-sim -i ../../temp_run_movement/ > ../../temp_run_movement/temp_run_movement.history")

    # 4) Read results
    movement = read_history("temp_run_movement.history")

    # 5) Delete files
    os.chdir("../../")
    os.system("rm -rf temp_run_movement")

    # 6) FFT On movement components
    frequency_comp = fftn(movement)
    return frequency_comp
