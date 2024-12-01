# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time

from kesslergame import Scenario, KesslerGame, GraphicsType

from src.fuzzy_thrust_controller import FuzzyThrustController
from src.fuzzy_thrust_controller_reversed import FuzzyThrustControllerReversed
from src.scott_dick_controller import ScottDickController
from test_controller import TestController
from graphics_both import GraphicsBoth
from src.range_fuzzy_reversed import FireRangerevController

# Define game scenario
my_test_scenario = Scenario(name='Test Scenario',
                            num_asteroids=15,
                            ship_states=[
                                {'position': (780, 220), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                {'position': (220, 220), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

# Define Game Settings
game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 1,
                 'graphics_obj': None,
                 'frequency': 30}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

# Evaluate the game
pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[FuzzyThrustControllerReversed(), FireRangerevController()])

# Print out some general info about the result
print('Scenario eval time: '+str(time.perf_counter()-pre))
print(score.stop_reason)
print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
print('Deaths: ' + str([team.deaths for team in score.teams]))
print('Accuracy: ' + str([team.accuracy for team in score.teams]))
print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))