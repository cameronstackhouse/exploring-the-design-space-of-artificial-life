"""
Module to get information from phenotypes for use in clustering, 
"""

from tokenize import String

def lz_phenotype(self) -> String:
        """
        Function to compress the phenotype produced by the CPPN using 
        lempel-ziv. The size of the generated compression can be used as a measurement
        of phenotypic complexity.

        :rtype string
        :return: lempel-ziv compressed string representation of the CPPN
        """
        #TODO lempel-ziv compression of phenotype. Does not work :( 
        #Gets phenotype of the CPPN genotype
        phenotype = self.to_phenotype()
        str_cells = [str(num) for num in phenotype]
        string_phenoype = "".join(str_cells)

        # Create codewords dictionary
        codewords = dict() # Codeword table
        word =  ""
        counter = 0
        for i in range(len(string_phenoype)):
            word += string_phenoype[i]
            if word not in codewords:
                codewords[word] = chr(counter)
                word = string_phenoype[i]
                counter+=1

        #Compress string phenotype using codewords and return 
        compressed = ""  
        word = ""
        for i in range(len(string_phenoype)):
            word += string_phenoype[i]
            if word not in codewords:
                compressed += codewords[word[:-1]]
                word = ""
        
        return compressed

def movement_frequency_components():
    pass

def motif_vectorisation():
    pass

def num_cells(phenotype) -> dict:
    """
    Function to return the number of the three different types of cells in the 
    phenotype

    :rtype: dict
    :return dictionary containing number of skin and muscle cells
    """
    none = 0
    skin = 0
    muscle = 0
    #Iterates through phenotype cells and increments the associated cell counter
    for cell in phenotype:
        if cell == 0:
            none+=1
        elif cell == 1:
            skin+=1
        elif cell == 2:
            muscle+=1
        
    return {"none": none, "skin": skin, "muscle": muscle}