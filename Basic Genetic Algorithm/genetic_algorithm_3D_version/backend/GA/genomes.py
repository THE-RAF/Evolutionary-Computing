import numpy as np
import random
from copy import deepcopy


class BaseGenome:
    def __init__(self, n_chromossomes, chromossome_size=None, set_genes=None):
        if not set_genes:
            self.chromossomes = [str().join([random.choice(['0', '1']) for _ in range(chromossome_size)]) for _ in range(n_chromossomes)]

        else:
            self.chromossomes = set_genes

    def evaluate(self, fitness_function):
        return fitness_function(self)

    @staticmethod
    def mutate_chromossome(chromossome):
        random_index = np.random.randint(0, len(chromossome))
        new_bit = '1' if chromossome[random_index] == '0' else '0'
        new_chromossome = chromossome[:random_index] + new_bit + chromossome[random_index+1:]

        return new_chromossome

    @staticmethod
    def chromossome_crossover(chromossome1, chromossome2):
        cutting_index = np.random.randint(1, len(chromossome1)-1)
        g1_1st_half, g1_2nd_half = chromossome1[:cutting_index], chromossome1[cutting_index:]
        g2_1st_half, g2_2nd_half = chromossome2[:cutting_index], chromossome2[cutting_index:]

        new_chromossome1 = g1_1st_half + g2_2nd_half
        new_chromossome2 = g2_1st_half + g1_2nd_half

        return new_chromossome1, new_chromossome2

    @staticmethod
    def mutate_genome(genome):
        new_genome = deepcopy(genome)
        for i, chromossome in enumerate(genome.chromossomes):
            new_genome.chromossomes[i] = BaseGenome.mutate_chromossome(chromossome)

        return new_genome

    @staticmethod
    def crossover_genome(genome1, genome2):
        new_genome1 = deepcopy(genome1)
        new_genome2 = deepcopy(genome2)

        for i, (chromossome1, chromossome2) in enumerate(zip(genome1.chromossomes, genome2.chromossomes)):
            new_chromossome1, new_chromossome2 = BaseGenome.chromossome_crossover(chromossome1, chromossome2)
            new_genome1.chromossomes[i] = new_chromossome1
            new_genome2.chromossomes[i] = new_chromossome2

        return new_genome1, new_genome2

    def __len__(self):
        return len(self.genes)

    def __str__(self):
        return str().join([chromossome + '\n' for chromossome in self.chromossomes])


def create_genome_class(n_chromossomes, chromossome_size):
    class Genome(BaseGenome):
        def __init__(self):
            BaseGenome.__init__(self,
                                n_chromossomes=n_chromossomes,
                                chromossome_size=chromossome_size,
                                set_genes=None
                                )

    return Genome
