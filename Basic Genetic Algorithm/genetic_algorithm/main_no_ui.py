import numpy as np

from backend.GA.genetic_algorithm import GeneticAlgorithm
from backend.GA.genomes import Genome1D
from backend.helpers import translate_bits_to_float


def fitness_function(genome):
    interval = [0, 512]
    x = translate_bits_to_float(bit_string=genome.genes[::-1], interval=interval)

    return abs(x*np.sin(np.sqrt(abs(x))))


ga = GeneticAlgorithm(
    pop_size=20,
    fitness_function=fitness_function,
    num_generations=5000,
    genome_type=Genome1D
    )

best_genome = ga.run()
results = {'best_fitness': ga.best_fitness, 'best_x': translate_bits_to_float(ga.best_genome.genes[::-1], [0, 512])}
print(results)
