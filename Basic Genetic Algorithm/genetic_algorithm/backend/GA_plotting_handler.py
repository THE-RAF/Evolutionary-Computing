import numpy as np

from backend.GA.genetic_algorithm import GeneticAlgorithm
from backend.GA.genomes import Genome1D
from backend.helpers import translate_bits_to_float


class GALivePlotHandler1D:
    def __init__(self,
        equation_string,
        interval,
        pop_size,
        crossover_rate,
        mutation_rate,
        num_generations,
        genome_size,
        ):

        self.equation_string = equation_string
        self.interval = interval

        self.GA = GeneticAlgorithm(
            pop_size=pop_size,
            fitness_function=lambda genome: self.fitness_function(genome, self.equation_string),
            genome_type=Genome1D,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            num_generations=num_generations,
            genome_size=genome_size,
            )

    def fitness_function(self, genome, equation_string):
        x = translate_bits_to_float(bit_string=genome.genes[::-1], interval=self.interval)

        return eval(equation_string)

    def get_xy_values_by_population(self, population):
        xs = []
        ys = []

        for genome in population:
            xs.append(translate_bits_to_float(bit_string=genome.genes[::-1], interval=self.interval))
            ys.append(self.GA.fitness_function(genome))

        return xs, ys

    def plot_GA(self,
        interval_upper_bound,
        interval_lower_bound,
        figure,
        canvas,
        timer,
        ):

        x_equation = np.linspace(interval_lower_bound, interval_upper_bound, num=300)
        y_equation = [eval(f'lambda x: {self.equation_string}')(x) for x in x_equation]

        current_population, done = self.GA.run_generation()
        x_genomes, y_genomes = self.get_xy_values_by_population(current_population)

        best_genome = self.GA.best_genome
        best_genome_x, best_genome_y = self.get_xy_values_by_population([best_genome])

        figure.clear()

        ax = figure.add_subplot(111)
        ax.plot(x_equation, y_equation, c='k', alpha=0.7, zorder=0)
        ax.scatter(x_genomes, y_genomes, c='#D60000', s=10, zorder=1)
        ax.scatter(best_genome_x, best_genome_y, c='#0063F2', s=15, zorder=1)

        canvas.draw()

        if done: timer.stop()
