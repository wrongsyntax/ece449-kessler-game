# scenario_test.py

import time
from kesslergame import Scenario, GraphicsType, TrainerEnvironment, KesslerGame

from ga_controller import FireRangerevController


def run_simulation(controller):
    """
    Run the simulation with the given controller and return a fitness score.

    :param controller: Instance of FuzzyThrustControllerReversed with specific parameters.
    :return: float representing the fitness score.
    """
    # Define game scenario
    my_test_scenario = Scenario(
        name='Test Scenario',
        num_asteroids=15,
        ship_states=[
            {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
        ],
        map_size=(1000, 800),
        time_limit=60,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

    # Define Game Settings with No Graphics for speed
    game_settings = {
        'perf_tracker': True,
        'graphics_type': GraphicsType.NoGraphics,  # Use NoGraphics for speed
        'realtime_multiplier': 0,  # Max-speed simulation
        'graphics_obj': None,
        'frequency': 30
    }

    # Choose the appropriate game environment
    game = TrainerEnvironment(settings=game_settings)

    # Evaluate the game
    pre = time.perf_counter()
    score, perf_data = game.run(scenario=my_test_scenario, controllers=[controller])
    post = time.perf_counter()

    # Compute fitness score based on performance metrics
    # Define your own weights and metrics as needed
    try:
        asteroids_hit = score.teams[0].asteroids_hit
        deaths = score.teams[0].deaths
        accuracy = score.teams[0].accuracy
        mean_eval_time = score.teams[0].mean_eval_time

        # Example fitness calculation
        fitness_score = (asteroids_hit * 10) - (deaths * 20) + (accuracy * 5) - (mean_eval_time * 1)
    except Exception as e:
        print(f"Error calculating fitness: {e}")
        fitness_score = 0  # Penalize failed simulations

    return fitness_score


if __name__ == "__main__":
    # Optional: Run a single simulation manually for testing
    controller = FireRangerevController()
    fitness = run_simulation(controller)
    print(f"Fitness Score: {fitness}")
