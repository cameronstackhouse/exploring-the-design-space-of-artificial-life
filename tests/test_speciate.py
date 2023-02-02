import os, sys
import numpy as np

#Adds parent directory to system path
p = os.path.abspath('.')
sys.path.insert(1, p)

from tools.speciate import speciate, share, distance
from networks import CPPN

def test_share_same():
    a = CPPN([8,8,7])

    print(distance(a, a))
