"""
Module to get information from phenotypes for use in clustering and comparing.
"""

import os
from scipy.fft import fftn # Fourier transform 
import numpy as np
from itertools import combinations, permutations
from read_files import read_history
from voxcraftpython.VoxcraftVXA import VXA
from voxcraftpython.VoxcraftVXD import VXD

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

def movement_frequency_components(CPPN) -> np.ndarray:
    """
    Function to get the frequency components of a xenobot movement path to use in clustering of xenobot behaviour.
    This is done using discrete 3-Dimensional Fourier transform on the X, Y, and Z movement coordinates of the 
    xenobot. 

    :param: CPPN which produces the xenobot
    :return: List of frequency components of the movement path of the xenobot, describing the behaviour of the xenobot
    """
    
    frequency_components = fftn(CPPN.movement)

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

def movement_components(cppn):
    """
    Gets the movement components of a xenobot
    from the history file produced by voxcraft-sim
    """
    # Make dir to run
    os.system("mkdir temp_run_movement")

    # 1) Make VXD File

    # 2) Make VXA File

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

if __name__ == "__main__":
    a = fftn([[1], [1]])

    print(a[0][0])

