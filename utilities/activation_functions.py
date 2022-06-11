import numpy as np
from math import pi

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

def mReLU(x: float) -> float:
    """

    """
    if x <= 0 and x > -1:
        return abs(x)
    elif x > 0 and x < 1:
        return 1 - x
    else:
        return 0 

def periodic(x: float) -> float:
    """
    Returns the value of the periodic function applied on x

    :param x: number to be passed into function
    :rtype: float
    :return: periodic function value of x
    """
    return (x+pi) % (2*pi) - pi

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

def tanh(x: float) -> float:
    """
    
    """
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def arctan(x: float) -> float:
    """
    
    """
    return np.arctan(x)

def relu(x: float) -> float:
    """
    
    """
    if x < 0:
        return 0
    else:
        return x

def sinusoid(x: float) -> float:
    """
    
    """
    return np.sin(x)

def sinc(x: float) -> float:
    """
    
    """
    if x == 0:
        return 1
    else:
        return (np.sin(x))/x