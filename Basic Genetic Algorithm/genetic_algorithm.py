import numpy as np
import random
from copy import deepcopy


class Genome:
    def __init__(self, size=10, set_genes=None):
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

        new_genome1 = Genome(set_genes=g1_1st_half + g2_2nd_half)
        new_genome2 = Genome(set_genes=g2_1st_half + g1_2nd_half)

        return new_genome1, new_genome2

    def __len__(self):
        return len(self.genes)

    def __str__(self):
        return self.genes


class GA:
    def __init__(self, pop_size, fitness_function, num_generations, crossover_rate=0.6, mutation_rate=0.01, genome_size=10):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.population = [Genome(size=genome_size) for _ in range(pop_size)]
        self.fitness_function = fitness_function

        self.best_fitness = -np.inf
        self.best_genome = None

        self.num_generations = num_generations
        self.current_generation = 1
        self.done = False

    @staticmethod
    def select(population, fitnesses, population_fitness):
        selection_probabilities = list(np.array(fitnesses) / population_fitness)
        selected_genomes = random.choices(population, weights=selection_probabilities, k=len(population))

        return selected_genomes

    def population_crossover(self, population):
        parents_half1 = population[:int(len(population)/2)]
        parents_half2 = population[int(len(population)/2):]

        offspring = []
        for parent1, parent2 in zip(parents_half1, parents_half2):
            if np.random.uniform() < self.crossover_rate:
                son1, son2 = Genome.crossover(parent1, parent2)
                offspring.append(son1)
                offspring.append(son2)

            else:
                offspring.append(parent1)
                offspring.append(parent2)

        return offspring

    def mutate_population(self, population):
        mutated_population = []
        for genome in population:
            if np.random.uniform() < self.mutation_rate:
                mutated_population.append(Genome.mutate(genome))

            else:
                mutated_population.append(genome)

        return mutated_population

    def run_generation(self):
        fitnesses = [genome.evaluate(self.fitness_function) for genome in self.population]
        population_fitness = sum(fitnesses)

        best_fitness_from_population = max(fitnesses)
        if best_fitness_from_population > self.best_fitness:
            self.best_fitness = best_fitness_from_population
            self.best_genome = self.population[np.argmax(fitnesses)]

        self.population = self.select(self.population, fitnesses, population_fitness)
        self.population = self.population_crossover(self.population)
        self.population = self.mutate_population(self.population)

        self.current_generation += 1
        self.done = self.current_generation >= self.num_generations

        return deepcopy(self.population), self.done

    def run(self, max_generations, fitness_threshold=200):
        while not self.done:
            population, self.done = self.run_generation()

        return self.best_genome


def translate_bits_to_float(bit_string, interval):
    step = (interval[1] - interval[0]) / (2**len(bit_string))
    x = step * sum([int(bit) * 2**i for i, bit in enumerate(bit_string)])

    return x

def fitness_function(genome):
    interval = [0, 512]
    x = translate_bits_to_float(bit_string=genome.genes[::-1], interval=interval)

    return abs(x*np.sin(np.sqrt(abs(x))))


if __name__ == '__main__':
    ga = GA(
        pop_size=20,
        fitness_function=fitness_function,
        num_generations=50000
        )

    best_genome = ga.run(max_generations=50000)
    results = {'best_fitness': ga.best_fitness, 'best_x': translate_bits_to_float(ga.best_genome.genes[::-1], [0, 512])}

    print(results)
