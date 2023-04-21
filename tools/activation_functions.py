"""
Module containing various different activation functions
"""

import numpy as np

np.seterr(divide="ignore", invalid="ignore") #Ignores the divide error as NaN already dealt with

def sigmoid(x: float) -> float:
    """
    Returns the value of the sigmoid function applied on x

    :param x: number to be passed into function
    :rtype: float
    :return: sigmoid function value of x
    """
    return 1 / (1 + np.exp(-x))

def neg_abs(x: float) -> float:
    """
    Returns the negative absolute value of the value
    passed into the function

    :param x:
    :rtype: float
    :return: negative absolute value of x
    """
    return np.nan_to_num(-(np.abs(x)))

def neg_square(x: float) -> float:
    """
    Returns the negative square root of the value
    passed into the function

    :param x:
    :rtype: float
    :return: negative square root of x
    """
    return np.nan_to_num(-(np.square(x)))

def sqrt_abs(x: float) -> float:
    """
    Returns the square root of the absolute value
    of the value passed into the function

    :param x:
    :rtype: float
    :return: square root of the absolute value of x
    """
    return np.nan_to_num(np.sqrt(np.abs(x)))

def neg_sqrt_abs(x: float) -> float:
    """
    Returns the value of the negative of the 
    square root of the absoulte value of the balue passed
    into the function

    :param x:
    :rtype: float
    :return: negative square root of the absolute value of x
    """
    return np.nan_to_num(-(sqrt_abs(x)))

def normalize(x: np.array) -> np.array:
    """
    Function to normalize a set of data to have values
    between -1 and 1

    :param x: array of input data to normalize
    :rtype: numpy array
    :return: normalized data (between -1 and 1)
    """
    x = x - np.min(x)
    x = x / np.max(x)
    x = np.nan_to_num(x)
    x *= 2
    x -= 1
    return x