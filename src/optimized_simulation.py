# optimized_simulation.py

import json
from fuzzy_thrust_controller_reversed import FuzzyThrustControllerReversed
from scenario_test import run_simulation
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

def load_parameters(json_file):
    """
    Load optimized parameters from a JSON file.
    :param json_file: Path to the JSON file.
    :return: Tuple of (thrust_params, turn_rate_params)
    """
    with open(json_file, 'r') as f:
        params = json.load(f)
    thrust_params = params['thrust_params']
    turn_rate_params = params['turn_rate_params']
    return thrust_params, turn_rate_params

def plot_membership_functions(params, variable_name):
    """
    Plot the trapezoidal membership functions for a given variable.
    :param params: Dictionary of membership functions and their boundaries.
    :param variable_name: Name of the variable (e.g., 'Thrust', 'Turn Rate').
    """
    universe = np.linspace(min([min(bounds) for bounds in params.values()]),
                           max([max(bounds) for bounds in params.values()]),
                           1000)
    plt.figure(figsize=(8, 6))
    for mf, bounds in params.items():
        mf_values = fuzz.trapmf(universe, bounds)
        plt.plot(universe, mf_values, label=mf)
    plt.title(f"{variable_name} Membership Functions")
    plt.xlabel(f"{variable_name}")
    plt.ylabel("Membership Degree")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Path to the best_parameters.json file
    PARAMS_FILE = 'best_parameters.json'

    # Load optimized parameters
    thrust_params, turn_rate_params = load_parameters(PARAMS_FILE)

    print("Best Thrust Parameters:")
    for mf, values in thrust_params.items():
        print(f"{mf}: {values}")

    print("\nBest Turn Rate Parameters:")
    for mf, values in turn_rate_params.items():
        print(f"{mf}: {values}")

    # Initialize the controller with optimized parameters
    optimized_controller = FuzzyThrustControllerReversed(
        thrust_params=thrust_params,
        turn_rate_params=turn_rate_params
    )

    print("\nOptimized Controller Initialized.")

    # Run simulation
    fitness_score = run_simulation(optimized_controller)
    print(f"\nOptimized Controller Fitness Score: {fitness_score}")

    # Optional: Visualize Thrust Membership Functions
    plot_membership_functions(thrust_params, 'Thrust')

    # Optional: Visualize Turn Rate Membership Functions
    plot_membership_functions(turn_rate_params, 'Turn Rate')

if __name__ == "__main__":
    main()
