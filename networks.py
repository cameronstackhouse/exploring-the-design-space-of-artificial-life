#CHECK WHAT ACTIVATION FUNCTIONS USED IN PAPER

#from utilities.activation_functions import sigmoid, symmetric, periodic, identity, gaussian, tanh, arctan, relu, sinusoid, sinc #Imports all defined activiation functions

"""
Module defining components for the creation of functioning
compositional pattern-producing networks
"""

class Node:
    """
    Class defining a node in a compositional pattern-producing network
    """
    def __init__(self, activation_function):
        """
        
        """
        self.activation_function = activation_function
        self.outputs = []


class CPPN:
    """
    Class defining a compositional pattern-producing network made of
    interconnected nodes with varying activation functions
    """
    def __init__(self):
        """
        
        """
        self.activation_functions = []


if __name__ == "__main__":
    """
    ******************
    DELETE FOR RELEASE
    TESTING ZONE
    ******************
    """
    