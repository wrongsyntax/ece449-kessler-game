# scenario_test.py

import time
import json

from kesslergame import Scenario, GraphicsType, TrainerEnvironment, KesslerGame

from ga_controller import FireRangerevController
# from scott_dick_controller import ScottDickController  # Uncomment if needed
from graphics_both import GraphicsBoth

# This is the code top open and load the best parameters inside best_parameters.json

with open('best_parameters.json', 'r') as f:
    optimized_params = json.load(f)
optimized_controller = FireRangerevController(
    thrust_params=optimized_params['thrust_params'],
    turn_rate_params=optimized_params['turn_rate_params']
)
print("Loaded optimized controller parameters.")


# Define game scenario
my_test_scenario = Scenario(
    name='Test Scenario',
    num_asteroids=15,
    ship_states=[
        {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
        # {'position': (600, 600), 'angle': 90, 'lives': 3, 'team': 2, 'mines_remaining': 3}
    ],
    map_size=(1000, 800),
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False
)

# Define Game Settings with Graphics Enabled
game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,  # Enabled graphics using Tkinter
    'realtime_multiplier': 1,  # Adjust as needed; set to 0 for max-speed simulation without delay
    'graphics_obj': None,
    'frequency': 30
}

# Choose the appropriate game environment
# For visualization (might slow down simulations)
game = KesslerGame(settings=game_settings)

# For max-speed, no-graphics simulation
# game = TrainerEnvironment(settings=game_settings)

# Evaluate the game
pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[optimized_controller])
post = time.perf_counter()

# Print out some general info about the result
print('Scenario eval time: {:.2f} seconds'.format(post - pre))
print('Scenario stop reason:', score.stop_reason)
print('Asteroids hit:', [team.asteroids_hit for team in score.teams])
print('Deaths:', [team.deaths for team in score.teams])
print('Accuracy:', [team.accuracy for team in score.teams])
print('Mean eval time:', [team.mean_eval_time for team in score.teams])
