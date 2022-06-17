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
    return np.nan_to_num(1/(1+np.exp(-x))) #Sigmoid function applied to x

def neg_abs(x: float) -> float:
    return np.nan_to_num(-(np.abs(x)))

def neg_square(x: float) -> float:
    return np.nan_to_num(-(np.square(x)))

def sqrt_abs(x: float) -> float:
    return np.nan_to_num(np.sqrt(np.abs(x)))

def neg_sqrt_abs(x: float) -> float:
    return np.nan_to_num(-(sqrt_abs(x)))