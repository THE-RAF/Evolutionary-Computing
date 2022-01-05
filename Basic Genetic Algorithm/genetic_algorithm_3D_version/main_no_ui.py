import numpy as np

from backend.GA.genetic_algorithm import GeneticAlgorithm
from backend.GA.genomes import create_genome_class
from backend.helpers import translate_bits_to_float




def fitness_function(genome):
    x_interval = [0, 4]
    y_interval = [0, 2]

    x = translate_bits_to_float(bit_string=genome.chromossomes[0][::-1], interval=x_interval)
    y = translate_bits_to_float(bit_string=genome.chromossomes[1][::-1], interval=y_interval)

    return 10 + x * np.sin(4*x) + 3 * np.sin(2*y)


ga_parameters = {
    'pop_size':        50  ,
    'num_generations': 5000,
    'crossover_rate':  0.6 ,
    'mutation_rate':   0.01,
    'chromossome_size': 10,
    'elitism_number': 1,
    'tournament_k': 2,
    'selection_method': 'roulette'
}

Genome2D = create_genome_class(n_chromossomes=2, chromossome_size=ga_parameters['chromossome_size'])

ga = GeneticAlgorithm(
    fitness_function=fitness_function,
    genome_type=Genome2D,
    parameters=ga_parameters,
    )

best_genome = ga.run()

intervals = [[0, 4], [0, 2]]
results = {'best_fitness': ga.best_fitness,
           'best_point': [translate_bits_to_float(chromossome[::-1], interval) for chromossome, interval in zip(ga.best_genome.chromossomes, intervals)]}
print(results)
