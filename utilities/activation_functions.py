import numpy as np

"""
Module containing various different activation functions
"""

def sigmoid(x: float) -> float:
    """
    Returns the value of the sigmoid function applied on x

    :param x: number to be passed into function
    :rtype: float
    :return: sigmoid function value of x
    """
    return 1/(1+np.exp(-x)) #Sigmoid function applied to x

def neg_abs(x: float) -> float:
    return -(np.abs(x))

def neg_square(x: float) -> float:
    return -(np.square(x))

def sqrt_abs(x: float) -> float:
    return np.abs(np.sqrt(x))

def neg_sqrt_abs(x: float) -> float:
    return -(sqrt_abs(x))