"""
Module to test the functionality of the read_settings.py module
using pytest.
"""

#TODO

import os, sys

#Adds parent directory to system path
p = os.path.abspath('.')
sys.path.insert(1, p)

from tools import read_outputs

def test_read_sim_output():
    """
    
    """
    output = read_outputs.read_sim_output("tools/example")
    assert output is not None 