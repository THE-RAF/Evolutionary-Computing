import numpy as np
import random
from copy import deepcopy


class GeneticAlgorithm:
    def __init__(self, fitness_function, genome_type, parameters):
        self.crossover_rate = parameters['crossover_rate']
        self.mutation_rate = parameters['mutation_rate']
        self.num_generations = parameters['num_generations']
        self.pop_size = parameters['pop_size']

        self.genome_type = genome_type

        self.population = [self.genome_type() for _ in range(parameters['pop_size'])]
        self.fitness_function = fitness_function

        self.best_fitness = -np.inf
        self.best_genome = None

        self.best_fitnesses = []
        self.mean_fitnesses = []

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
                son1, son2 = self.genome_type.crossover_genome(parent1, parent2)
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
                mutated_population.append(self.genome_type.mutate_genome(genome))

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

        self.best_fitnesses.append(self.best_fitness)
        self.mean_fitnesses.append(population_fitness/self.pop_size)

        self.current_generation += 1
        self.done = self.current_generation >= self.num_generations

        return deepcopy(self.population), self.done

    def run(self):
        while not self.done:
            population, self.done = self.run_generation()

        return self.best_genome
