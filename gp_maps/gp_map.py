import pickle
#TODO Use graphviz

class GenotypePhenotypeMap:
    """
    
    """

    def __init__(self, num_genotypes) -> None:
        self.gp_map = {}
        #NOTE Probably mapped phenotype: [genotypes]

        #TODO Generate Genotypes
        genotypes = 0
        while genotypes < num_genotypes:
            pass

        #TODO Map neutral network

        #TODO Map each Genotype to Phenotype


class FitnessLandscape:
    """
    
    """
    def __init__(self, num_genotypes) -> None:
        self.gp_map = GenotypePhenotypeMap(num_genotypes) #Generates genotype-phenotype map
        self.phenotype_fitness_map = {} #NOTE Probably mapped fitness: [phenotypes] #Generates phenotype-fitness map

        for key in self.gp_map:
            phenotype_fitness = self.gp_map[key][0].fitness

            if phenotype_fitness in self.phenotype_fitness_map.keys():
                self.phenotype_fitness_map[phenotype_fitness].append(key)
            else:
                self.phenotype_fitness_map[phenotype_fitness] = [key]
