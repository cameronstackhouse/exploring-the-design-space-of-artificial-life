import os, sys

#Adds parent directory to system path
p = os.path.abspath('.')
sys.path.insert(1, p)

import tools.phenotype_information

def test_lempel_ziv() -> None:
    """
    
    """
    output, d = tools.phenotype_information.lz_phenotype("101010101")
    print(d)


test_lempel_ziv()