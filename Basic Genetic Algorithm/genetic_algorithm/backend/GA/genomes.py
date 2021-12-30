import numpy as np
import random
from copy import deepcopy


class Genome1D:
    def __init__(self, size=None, set_genes=None):
        if not set_genes:
            self.genes = str().join([random.choice(['0', '1']) for _ in range(size)])

        else:
            self.genes = set_genes

    def evaluate(self, fitness_function):
        return fitness_function(self)

    @staticmethod
    def mutate(genome):
        new_genome = deepcopy(genome)
        random_index = np.random.randint(0, len(genome))
        new_bit = '1' if genome.genes[random_index] == '0' else '0'
        new_genome.genes = genome.genes[:random_index] + new_bit + genome.genes[random_index+1:]

        return new_genome

    @staticmethod
    def crossover(genome1, genome2):
        cutting_index = np.random.randint(1, len(genome1)-1)
        g1_1st_half, g1_2nd_half = genome1.genes[:cutting_index], genome1.genes[cutting_index:]
        g2_1st_half, g2_2nd_half = genome2.genes[:cutting_index], genome2.genes[cutting_index:]

        new_genome1 = Genome1D(set_genes=g1_1st_half + g2_2nd_half)
        new_genome2 = Genome1D(set_genes=g2_1st_half + g1_2nd_half)

        return new_genome1, new_genome2

    def __len__(self):
        return len(self.genes)

    def __str__(self):
        return self.genes
