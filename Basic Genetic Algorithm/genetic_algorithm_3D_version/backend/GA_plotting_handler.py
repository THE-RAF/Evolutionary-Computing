import matplotlib.pyplot as plt
import numpy as np

from backend.GA.genetic_algorithm import GeneticAlgorithm
from backend.GA.genomes import create_genome_class
from backend.helpers import translate_bits_to_float


class GALivePlotHandler1D:
    def __init__(self,
                 equation_string,
                 interval,
                 parameters,
                 ):

        self.equation_string = equation_string
        self.interval = interval

        self.GA = GeneticAlgorithm(
            fitness_function=lambda genome: self.fitness_function(genome, self.equation_string),
            genome_type=create_genome_class(n_chromossomes=1, chromossome_size=10),
            parameters=parameters,
            )

    def fitness_function(self, genome, equation_string):
        x = translate_bits_to_float(bit_string=genome.chromossomes[0][::-1], interval=self.interval)

        return eval(equation_string)

    def get_xy_values_by_population(self, population):
        xs = []
        ys = []

        for genome in population:
            xs.append(translate_bits_to_float(bit_string=genome.chromossomes[0][::-1], interval=self.interval))
            ys.append(self.GA.fitness_function(genome))

        return xs, ys

    def get_best_xy_values(self):
        best_genome = self.GA.best_genome
        x = translate_bits_to_float(bit_string=best_genome.chromossomes[0][::-1], interval=self.interval)
        y = self.GA.fitness_function(best_genome)

        return x, y

    def plot_GA(self,
                figure,
                canvas,
                timer,
                ):

        x_equation = np.linspace(self.interval[0], self.interval[1], num=300)
        y_equation = [eval(f'lambda x: {self.equation_string}')(x) for x in x_equation]

        current_population, done = self.GA.run_generation()
        x_genomes, y_genomes = self.get_xy_values_by_population(current_population)

        best_genome = self.GA.best_genome
        best_genome_x, best_genome_y = self.get_xy_values_by_population([best_genome])

        figure.clear()

        ax_GA = figure.add_subplot(121)
        ax_GA.plot(x_equation, y_equation, c='k', alpha=0.7, zorder=0)
        ax_GA.scatter(x_genomes, y_genomes, c='#f17587', s=10, zorder=1)
        ax_GA.scatter(best_genome_x, best_genome_y, c='#77cdf2', s=15, zorder=1)

        ax_performance = figure.add_subplot(122)
        ax_performance.plot(self.GA.best_fitnesses, c='#77cdf2', label='Best fitness')
        ax_performance.plot(self.GA.mean_fitnesses, c='#f17587', label='Mean fitness')

        ax_performance.set_xlabel('Generations')
        ax_performance.set_ylabel('Fitness')
        ax_performance.legend()

        plt.tight_layout()
        canvas.draw()

        if done: timer.stop()


class GALivePlotHandler2D:
    def __init__(self,
                 equation_string,
                 intervals,
                 parameters,
                 ):
        self.equation_string = equation_string
        self.intervals = intervals

        self.GA = GeneticAlgorithm(
            fitness_function=lambda genome: self.fitness_function(genome, self.equation_string),
            genome_type=create_genome_class(n_chromossomes=2, chromossome_size=parameters['chromossome_size']),
            parameters=parameters,
        )

    def fitness_function(self, genome, equation_string):
        x = translate_bits_to_float(bit_string=genome.chromossomes[0][::-1], interval=self.intervals[0])
        y = translate_bits_to_float(bit_string=genome.chromossomes[1][::-1], interval=self.intervals[1])

        return eval(equation_string)

    def get_xyz_values_by_population(self, population):
        xs = []
        ys = []
        zs = []

        for genome in population:
            xs.append(translate_bits_to_float(bit_string=genome.chromossomes[0][::-1], interval=self.intervals[0]))
            ys.append(translate_bits_to_float(bit_string=genome.chromossomes[1][::-1], interval=self.intervals[1]))
            zs.append(self.GA.fitness_function(genome))

        return xs, ys, zs

    def get_best_xyz_values(self):
        best_genome = self.GA.best_genome
        x = translate_bits_to_float(bit_string=best_genome.chromossomes[0][::-1], interval=self.intervals[0])
        y = translate_bits_to_float(bit_string=best_genome.chromossomes[1][::-1], interval=self.intervals[1])
        z = self.GA.fitness_function(best_genome)

        return x, y, z

    def plot_GA(self,
                figure,
                canvas,
                timer,
                ):

        x = np.arange(self.intervals[0][0], self.intervals[0][1], 0.1)
        y = np.arange(self.intervals[1][0], self.intervals[1][1], 0.1)

        x, y = np.meshgrid(x, y)
        z = eval(self.equation_string)

        current_population, done = self.GA.run_generation()
        x_genomes, y_genomes, z_genomes = self.get_xyz_values_by_population(current_population)

        best_genome = self.GA.best_genome
        best_genome_x, best_genome_y, best_genome_z = self.get_xyz_values_by_population([best_genome])

        figure.clear()

        ax_GA = figure.add_subplot(121, projection='3d', zorder=0)
        ax_GA.plot_wireframe(x, y, z, color='k', zorder=0, alpha=0.3)

        red_hexcode = '#f4404c'
        blue_hexcode = '#0000ff'
        light_blue_hexcode = '#77cdf2'

        ax_GA.scatter(x_genomes, y_genomes, z_genomes, c=red_hexcode, s=10, zorder=0)
        ax_GA.scatter(best_genome_x, best_genome_y, best_genome_z, c=blue_hexcode, s=15, zorder=1)

        ax_performance = figure.add_subplot(122)
        ax_performance.plot(self.GA.best_fitnesses, c=light_blue_hexcode, label='Best fitness')
        ax_performance.plot(self.GA.mean_fitnesses, c=red_hexcode, label='Mean fitness')

        ax_performance.set_xlabel('Generations')
        ax_performance.set_ylabel('Fitness')
        ax_performance.legend()

        plt.tight_layout()
        canvas.draw()

        if done: timer.stop()
