import pickle

class GenotypePhenotypeMap:
    """
    
    """

    def __init__(self, genotypes, phenotypes, title) -> None:
        self.gp_map = {}
        self.phenotype_fitness_map = {}
        self.title = title

        for genotype in genotypes:
            phenotype = genotype.to_phenotype()
            #TODO Map to phenotype

        #TODO Map phenotype to fitness

    def draw() -> None:
        #TODO
        pass