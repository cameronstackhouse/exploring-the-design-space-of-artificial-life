import os, sys
import numpy as np

#Adds parent directory to system path
p = os.path.abspath('.')
sys.path.insert(1, p)

from tools.speciate import speciate, share, distance
from cppn_neat import add_node
from networks import CPPN

def test_distance_same():
    a = CPPN([8,8,7])

    assert distance(a, a) == 0

def test_distance_different():
    a = CPPN([8,8,7])
    b = CPPN([8,8,7])

    assert distance(a, b) > 0

def test_distance_one_bigger():
    a = CPPN([8,8,7])
    bigger = CPPN([8,8,7])

    for _ in range(100):
        add_node(bigger)
    
    num_nodes = 0
    for layer in bigger.nodes:
        num_nodes += len(layer)

    assert num_nodes == 107
    assert share(distance(a, bigger), 1.0) == 0