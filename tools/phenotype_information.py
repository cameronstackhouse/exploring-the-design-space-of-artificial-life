"""
Module to get information from phenotypes for use in clustering and comparing.
"""

from scipy.fft import fftn # Fourier transform 
import numpy as np

def lz_phenotype(phenotype) -> str:
    """
    Function to compress the phenotype produced by the CPPN using 
    lempel-ziv. The size of the generated compression can be used as a measurement
    of phenotypic complexity.

    :rtype string
    :return: lempel-ziv compressed string representation of the CPPN
    """
    pass

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
    #TODO
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