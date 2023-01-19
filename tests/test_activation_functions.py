"""
Module testing the list of defined activation functions in the module
"activation_functions.py" to ensure they produce the correct outputs
"""

import os, sys
import numpy as np

#Adds parent directory to system path
p = os.path.abspath('.')
sys.path.insert(1, p)

from tools.activation_functions import sigmoid, neg_abs, neg_square, sqrt_abs, neg_sqrt_abs, normalize #Imports all activation functions

def test_sigmoid() -> None:
    """
    Tests the sigmoid function to ensure it returns the correct
    values
    """
    assert round(sigmoid(0.5), 3) == 0.622
    assert round(sigmoid(-0.5), 3) == 0.378
    assert round(sigmoid(20), 3) == 1

def test_neg_abs() -> None:
    """
    Tests the negative absolute function to ensure it returns the 
    correct values
    """
    assert neg_abs(-1) == -1
    assert neg_abs(1) == -1
    assert neg_abs(0) == 0

def test_neg_square() -> None:
    """
    Tests the negative square function to ensure it returns the 
    correct values
    """
    assert round(neg_square(2), 3) == -4
    assert round(neg_square(-4), 3) == -16
    assert round(neg_square(0), 3) == 0 

def test_sqrt_abs() -> None:
    """
    Tests the square root absolute function to ensure it returns the 
    correct values
    """
    assert round(sqrt_abs(2), 3) == 1.414
    assert round(sqrt_abs(-6), 3) == 2.449
    assert round(sqrt_abs(-4), 3) == 2

def test_neg_sqrt_abs() -> None:
    """
    Tests the negative square root absolute function to ensure it returns 
    the correct values
    """
    assert round(neg_sqrt_abs(2), 3) == -1.414
    assert round(neg_sqrt_abs(-6), 3) == -2.449
    assert round(neg_sqrt_abs(-4), 3) == -2

def test_normalize() -> None:
    """
    Tests the normalize function to ensure it returns the 
    correct values
    """
    x = np.array([1,2,3,4,5,6])
    normalized = normalize(x) # Normalized 1D array

    # Ensures all numbers in new normalized array are between -1 and 1
    for number in normalized:
        assert number >= -1 and number <= 1
    
    # Asserts that the length of the normalized array has not changed
    # and that the previous maximum and minimum values have been converted
    # to 1 and -1 respectivley
    assert len(normalized) == 6
    assert normalized[5] == 1
    assert normalized[0] == -1

    # Ensures the function works on 2D arrays
    y = np.array([[1,2,3], [4,5,6]])
    normalized_y = normalize(y)

    # Ensures all numbers in new normalized array are between -1 and 1
    for row in normalized_y:
        for number in row:
            assert number >= -1 and number <= 1           



