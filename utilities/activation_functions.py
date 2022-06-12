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

def periodic(x: float) -> float:
    """
    Returns the value of the periodic function applied on x

    :param x: number to be passed into function
    :rtype: float
    :return: periodic function value of x
    """
    return np.sin(x) #Periodic sin function applied to x

def identity(x: float) -> float:
    """
    Returns the identify function applied to x

    :param x: number to be passed into the function
    :rtype: float
    :return: identity function value of x
    """
    return x

def gaussian(x: float) -> float:
    """
    Returns the value of the gaussian function applied on x

    :param x: number to be passed into function
    :rtype: float
    :return: sigmoid function value of x
    """
    return np.exp((-x)**2)

def repeat_asym(x: float) -> float:
    """
    PLACEHOLDER

    :param x: number to be passed into function
    :rtype: float
    :return: 
    """
    return x % 1

def absolute(x: float) -> float:
    """
    Returns the absolute value of x

    :param x: number to be passed into function
    :rtype: float
    :return: absolute value of x
    """
    return abs(x)

def inverse(x: float) -> float:
    """
    Returns the inverse value of x

    :param x: number to be passed into function
    :rtype: float
    :return: inverse value of x
    """
    return -x

def symmetric(x: float) -> float:
    """
    Returns the value of x when a symmetric function is applied on it

    :param x: number to be passed into function
    :rtype: float
    :return: symmetric function value when applied with x
    """
    if x > 1 or x < -1: #If x is out of range then return 0
        return 0
    elif x <= 0: #If x is less than 0 and in range then return the absolute value of x
        return abs(x)
    else: #Else if x is in range and less than or equal to 1 then return 1 - x
        return 1 - x