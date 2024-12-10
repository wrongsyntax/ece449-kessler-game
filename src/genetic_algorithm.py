import random
import json
import time
from deap import base, creator, tools, algorithms
import numpy as np
from ga_controller import FireRangerevController
from ga_scenario_test import run_simulation
from typing import List, Tuple, Dict
from joblib import Parallel, delayed


def mate_and_sort(ind1, ind2):
    """
    Custom crossover that performs two-point crossover and sorts parameters for each MF.
    """
    # Perform a standard two-point crossover
    tools.cxTwoPoint(ind1, ind2)

    # Define the number of genes per membership function
    genes_per_mf = 4
    total_mfs = 10  # 5 thrust + 5 turn_rate membership classes

    for i in range(total_mfs):
        start = i * genes_per_mf
        end = start + genes_per_mf
        ind1[start:end] = sorted(ind1[start:end])
        ind2[start:end] = sorted(ind2[start:end])

    return ind1, ind2

def mutate_and_sort(individual, mu=0, sigma=10, indpb=0.2):
    """
    Custom mutation that applies Gaussian mutation and sorts parameters for each MF.
    """

    tools.mutGaussian(individual, mu, sigma, indpb)

    # Define the number of genes per membership function
    genes_per_mf = 4
    total_mfs = 10  # 5 thrust + 5 turn_rate

    for i in range(total_mfs):
        start = i * genes_per_mf
        end = start + genes_per_mf
        individual[start:end] = sorted(individual[start:end])

    return individual,

# Main function that runs the genetic algorithm
class GeneticAlgorithm:
    def __init__(self, population_size=3, generations=4, crossover_prob=0.7, mutation_prob=0.2):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # Define the ranges for each parameter
        # Adjust these ranges based on domain knowledge
        self.param_ranges = {
            'thrust': {
                'reverse_full': {'min1': -480, 'max1': -480, 'min2': -300, 'max2': -200},
                'reverse_medium': {'min1': -300, 'max1': -200, 'min2': -100, 'max2': -5},
                'coast': {'min1': -100, 'max1': -5, 'min2': 5, 'max2': 100},
                'forward_medium': {'min1': 5, 'max1': 100, 'min2': 200, 'max2': 300},
                'forward_full': {'min1': 200, 'max1': 300, 'min2': 480, 'max2': 480}
            },
            'turn_rate': {
                'sharp_left': {'min1': -180, 'max1': -180, 'min2': -120, 'max2': -90},
                'left': {'min1': -120, 'max1': -90, 'min2': -30, 'max2': -10},
                'straight': {'min1': -30, 'max1': -10, 'min2': 10, 'max2': 30},
                'right': {'min1': 10, 'max1': 30, 'min2': 90, 'max2': 120},
                'sharp_right': {'min1': 90, 'max1': 120, 'min2': 180, 'max2': 180}
            }
        }

        # Flatten all parameter ranges into a list of tuples (min, max)
        self.all_param_ranges = []
        for category, mfs in self.param_ranges.items():
            for mf, ranges in mfs.items():
                # Each trapezoidal MF has 4 parameters
                self.all_param_ranges.append((ranges['min1'], ranges['max1']))  # p1
                self.all_param_ranges.append((ranges['min1'], ranges['max1']))  # p2
                self.all_param_ranges.append((ranges['min2'], ranges['max2']))  # p3
                self.all_param_ranges.append((ranges['min2'], ranges['max2']))  # p4

        # Fitness framework
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Required tool box attributes
        self.toolbox.register("individual", self.initIndividual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", mate_and_sort)
        self.toolbox.register("mutate", mutate_and_sort, mu=0, sigma=10, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.fitness)

    def initIndividual(self):
        """
        Initialize an individual with random parameters within the ranges.
        """
        individual = []
        for (min_val, max_val) in self.all_param_ranges:
            gene = random.uniform(min_val, max_val)
            individual.append(gene)
        return creator.Individual(individual)

    def decode_individual(self, individual: List[float]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Decode the individual list into thrust and turn_rate parameter dictionaries.
        Each set of four parameters is sorted to satisfy a <= b <= c <= d.
        """
        thrust_params = {}
        turn_rate_params = {}
        idx = 0
        for mf in ['reverse_full', 'reverse_medium', 'coast', 'forward_medium', 'forward_full']:
            # Sort each set of four parameters
            thrust_params[mf] = sorted(individual[idx:idx + 4])
            idx += 4
        for mf in ['sharp_left', 'left', 'straight', 'right', 'sharp_right']:
            # Sort each set of four parameters
            turn_rate_params[mf] = sorted(individual[idx:idx + 4])
            idx += 4
        return thrust_params, turn_rate_params

    def fitness(self, individual: List[float]) -> Tuple[float,]:
        """
        Fitness function evaluating how well parameters oerformed based on (asteroids_hit * 10) - (deaths * 20) + (accuracy * 5) - (mean_eval_time * 1)
        """
        thrust_params, turn_rate_params = self.decode_individual(individual)

        # Initialize the controller with these parameters
        controller = FireRangerevController(thrust_params=thrust_params, turn_rate_params=turn_rate_params)

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
        """
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)  # Keep track of the best individual

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Evaluating fitness in parallel for accelerated training
        def evaluate_population(evaluate_func, population):
            return Parallel(n_jobs=-1)(delayed(evaluate_func)(ind) for ind in population)

        # Replace DEAP's map with the evaluate_population function
        self.toolbox.register("map", evaluate_population)

        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob,
                                       ngen=self.generations, stats=stats, halloffame=hof, verbose=True)

        best_individual = hof[0]
        best_params = self.decode_individual(best_individual)
        print("Best Individual:", best_individual)
        print("Best Parameters:", best_params)

        return best_params

    def save_best_parameters(self, best_params: Tuple[Dict[str, List[float]], Dict[str, List[float]]],
                             filename='best_parameters.json'):
        for param_type in ['thrust_params', 'turn_rate_params']:
            for mf in best_params[0] if param_type == 'thrust_params' else best_params[1]:
                best_params[0][mf] = sorted(best_params[0][mf]) if param_type == 'thrust_params' else sorted(
                    best_params[1][mf])

        with open(filename, 'w') as f:
            json.dump({
                'thrust_params': best_params[0],
                'turn_rate_params': best_params[1]
            }, f, indent=4)

        print(f"Best parameters saved to '{filename}'.")

def run_ga_optimization():
    ga = GeneticAlgorithm(population_size=2, generations=3, crossover_prob=0.9, mutation_prob=0.1)
    start_time = time.time()
    best_params = ga.run()
    print(f"Optimization took {time.time() - start_time:.2f} seconds.")
    ga.save_best_parameters(best_params)
    return best_params

if __name__ == "__main__":
    best_parameters = run_ga_optimization()
    print("Optimization complete. Best parameters saved to 'best_parameters.json'.")
