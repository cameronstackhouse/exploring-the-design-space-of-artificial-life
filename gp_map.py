import pickle

class Geneotype:
    def __init__(self, geneome) -> None:
        self.genome = geneome

class GenotypePhenotypeMap:
    def __init__(self) -> None:
        self.map = {}
        self.phenotype_fitness = {}
    
    def gen_genotypes_of_n(self, n: int):
        pass

    def draw(self):
        pass

class MultiLayeredGenotypePhenotypeMap:
    def __init__(self) -> None:
        self.map = {}

def save(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file)

def load(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
        return obj

gp = GenotypePhenotypeMap

save(gp, "test")

new = load("test")

print(gp.map)