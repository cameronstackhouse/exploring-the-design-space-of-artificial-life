"""
Module containing fitness functions for use in voxcraft-sim

NOTE: Variables to play around with
- x
- y
- z
- hit
- t
- angle
- closeness
- numClosePairs
- num_voxel
"""

from enum import Enum
from lxml import etree

class FitnessFunction(Enum):
    """
    List of fitness functions to test a population for fitness
    """
    MAX_DISTANCE = 1
    OBJECT_EXPULSION = 2
    ABS_DISTANCE = 3
    X_ONLY = 4
    Y_ONLY = 5
    #TODO Add more fitness func values here

def max_distance(fitness: etree.ElementTree) -> etree.ElementTree:
    """ 
    
    """
    sqrt = etree.SubElement(fitness, "mtSQRT")
    add = etree.SubElement(sqrt, "mtADD")
    mul = etree.SubElement(add, 'mtMUL')
    etree.SubElement(mul, "mtVAR").text = 'x'
    etree.SubElement(mul, "mtVAR").text = 'x'
    mul2 = etree.SubElement(add, 'mtMUL')
    etree.SubElement(mul2, "mtVAR").text = 'y'
    etree.SubElement(mul2, "mtVAR").text = 'y'

def abs_distance(fitness: etree.ElementTree) -> etree.ElementTree:
    """ 
    
    """
    add = etree.SubElement(fitness, "mtADD")
    
    abs = etree.SubElement(add, "mtABS")
    abs2 = etree.SubElement(add, "mtABS")
    abs3 = etree.SubElement(add, "mtABS")
    
    etree.SubElement(abs, "mtVAR").text = 'x'
    etree.SubElement(abs2, "mtVAR").text = 'y'
    etree.SubElement(abs3, "mtVAR").text = "z"

def x_only(fitness: etree.ElementTree) -> etree.ElementTree:
    """ 
    
    """
    abs = etree.SubElement(fitness, "mtABS")
    etree.SubElement(abs, "mtVAR").text = 'x'

def y_only(fitness: etree.ElementTree) -> etree.ElementTree:
    """
     
    """
    abs = etree.SubElement(fitness, "mtABS")
    etree.SubElement(abs, "mtVAR").text = 'y'

def penalise_large_total_distance(fitness: etree.ElementTree) -> etree.ElementTree:
    """
    Fitness function which penalises the fitness of
    xenobots which are too large
    """
    pass

def tall_obsticle(tree: etree.ElementTree) -> etree.ElementTree:
    pass