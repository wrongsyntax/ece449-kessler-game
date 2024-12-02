# genetic_algorithm.py

import random
import json
from deap import base, creator, tools, algorithms
import numpy as np
from fuzzy_thrust_controller_reversed import FuzzyThrustControllerReversed
from scenario_test import run_simulation
from typing import List, Tuple, Dict

class GeneticAlgorithm:
    def __init__(self, population_size=20, generations=30, crossover_prob=0.7, mutation_prob=0.2):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # Define the ranges for each parameter
        # Assuming each membership function boundary can vary within certain limits
        # Adjust these ranges based on domain knowledge
        self.param_ranges = {
            'thrust': {
                'reverse_full': {'min1': -150, 'max1': -50, 'min2': -150, 'max2': -50},
                'reverse_medium': {'min1': -100, 'max1': -30, 'min2': -30, 'max2': 0},
                'coast': {'min1': -20, 'max1': -1, 'min2': 1, 'max2': 20},
                'forward_medium': {'min1': 0, 'max1': 30, 'min2': 30, 'max2': 80},
                'forward_full': {'min1': 40, 'max1': 150, 'min2': 150, 'max2': 150}
            },
            'turn_rate': {
                'sharp_left': {'min1': -180, 'max1': -100, 'min2': -180, 'max2': -100},
                'left': {'min1': -150, 'max1': -50, 'min2': -150, 'max2': -50},
                'straight': {'min1': -100, 'max1': -10, 'min2': 10, 'max2': 100},
                'right': {'min1': 50, 'max1': 150, 'min2': 150, 'max2': 150},
                'sharp_right': {'min1': 100, 'max1': 180, 'min2': 180, 'max2': 180}
            }
        }

        # Flatten all parameter ranges into a list of tuples (min, max)
        self.all_param_ranges = []
        for category, mfs in self.param_ranges.items():
            for mf, ranges in mfs.items():
                # Each trapezoidal MF has 4 parameters
                self.all_param_ranges.append( (ranges['min1'], ranges['max1']) )  # p1
                self.all_param_ranges.append( (ranges['min1'], ranges['max1']) )  # p2
                self.all_param_ranges.append( (ranges['min2'], ranges['max2']) )  # p3
                self.all_param_ranges.append( (ranges['min2'], ranges['max2']) )  # p4

        # Setup DEAP framework
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizing fitness
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Define 'initIndividual' method
        self.toolbox.register("individual", self.initIndividual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.fitness)

    def initIndividual(self):
        """
        Initialize an individual with random parameters within the specified ranges.
        :return: An Individual instance.
        """
        individual = []
        for (min_val, max_val) in self.all_param_ranges:
            gene = random.uniform(min_val, max_val)
            individual.append(gene)
        return creator.Individual(individual)

    def decode_individual(self, individual: List[float]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Decode the individual list into thrust and turn_rate parameter dictionaries.
        :param individual: List of floats representing the individual's genes.
        :return: Tuple of (thrust_params, turn_rate_params)
        """
        thrust_params = {}
        turn_rate_params = {}
        idx = 0
        for mf in ['reverse_full', 'reverse_medium', 'coast', 'forward_medium', 'forward_full']:
            thrust_params[mf] = individual[idx:idx+4]
            idx += 4
        for mf in ['sharp_left', 'left', 'straight', 'right', 'sharp_right']:
            turn_rate_params[mf] = individual[idx:idx+4]
            idx += 4
        return thrust_params, turn_rate_params

    def fitness(self, individual: List[float]) -> Tuple[float,]:
        """
        Fitness function that evaluates how well the controller with given parameters performs.
        :param individual: List of parameters representing membership function boundaries.
        :return: Tuple containing the fitness score.
        """
        thrust_params, turn_rate_params = self.decode_individual(individual)

        # Initialize the controller with these parameters
        controller = FuzzyThrustControllerReversed(thrust_params=thrust_params, turn_rate_params=turn_rate_params)

        # Run the game simulation using scenario_test.py's run_simulation function
        try:
            fitness_score = run_simulation(controller)
        except Exception as e:
            print(f"Simulation failed: {e}")
            fitness_score = 0  # Penalize failed simulations

        return (fitness_score,)

    def run(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Run the genetic algorithm optimization process.
        :return: Best found parameters as a tuple (thrust_params, turn_rate_params).
        """
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)  # Keep track of the best individual

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob,
                                       ngen=self.generations, stats=stats, halloffame=hof, verbose=True)

        best_individual = hof[0]
        best_params = self.decode_individual(best_individual)
        print("Best Individual:", best_individual)
        print("Best Parameters:", best_params)

        return best_params

    def save_best_parameters(self, best_params: Tuple[Dict[str, List[float]], Dict[str, List[float]]], filename='best_parameters.json'):
        """
        Save the best parameters to a JSON file.
        :param best_params: Tuple of (thrust_params, turn_rate_params)
        :param filename: Name of the JSON file to save the parameters.
        """
        with open(filename, 'w') as f:
            json.dump({
                'thrust_params': best_params[0],
                'turn_rate_params': best_params[1]
            }, f, indent=4)

def run_ga_optimization():
    ga = GeneticAlgorithm(population_size=20, generations=30, crossover_prob=0.7, mutation_prob=0.2)
    best_params = ga.run()
    ga.save_best_parameters(best_params)
    return best_params

if __name__ == "__main__":
    # Run the GA to optimize the controller
    best_parameters = run_ga_optimization()
    print("Optimization complete. Best parameters saved to 'best_parameters.json'.")
