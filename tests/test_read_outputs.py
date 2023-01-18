"""
Module to test the functionality of the read_outputs.py module
using pytest.
"""

import os, sys

#Adds parent directory to system path
p = os.path.abspath('.')
sys.path.insert(1, p)

from tools import read_files

def test_read_sim_output() -> None:
    """
    Function to test the read_sim_output funtion from the read_outputs.py
    module.
    """
    output = read_files.read_sim_output("tools/example")
    assert output is not None

def test_read_settings() -> None:
    """
    Function to test the read_settings function from the read_outputs.py
    module.
    """
    a = read_files.read_settings("settings")
    assert a is not None