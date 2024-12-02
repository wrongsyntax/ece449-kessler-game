from time import time_ns
from typing import Dict, Tuple
import EasyGA
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kesslergame import KesslerController
from kesslergame import Scenario, TrainerEnvironment, GraphicsType

def normalize_angle(angle):
    """Normalize angle to [0, 360] range"""
    return angle % 360


# Fitness function for GA
def fitness_function(chromosome):
    """
    Evaluates the fitness of a chromosome by running a game simulation
    and calculating the number of asteroids hit.
    """
    thrust_params = chromosome[0].value
    turn_rate_params = chromosome[1].value

    # Set up the controller using the chromosome parameters
    controller = FuzzyThrustControllerReversed(thrust_params, turn_rate_params)

    # Define the game scenario
    scenario = Scenario(
        name="GA Fitness Scenario",
        num_asteroids=15,
        ship_states=[{'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3}],
        map_size=(1000, 800),
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

    # Run simulation
    game_settings = {'perf_tracker': True, 'graphics_type': None, 'realtime_multiplier': 0, 'frequency': 30}
    game = TrainerEnvironment(settings=game_settings)
    score, _ = game.run(scenario=scenario, controllers=[controller])

    # Fitness is the number of asteroids hit (maximize this)
    return -score.teams[0].asteroids_hit  # Negative for minimization


class FuzzyThrustControllerReversed(KesslerController):
    def __init__(self, thrust_params=None, turn_rate_params=None):
        self.danger_ctrl = None
        self.control_ctrl = None
        self.eval_frames = 0
        self.thrust_params = thrust_params
        self.turn_rate_params = turn_rate_params

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        if self.danger_ctrl is None:
            self.danger_ctrl = self.init_danger_fs()
        if self.control_ctrl is None:
            self.control_ctrl = self.init_control_fs(ship_state)

        closest_features = self.calculate_closest_asteroid_features(ship_state, game_state)
        asteroid_angle = normalize_angle(np.degrees(closest_features['relative_angle']))
        attack_heading = normalize_angle(asteroid_angle)
        current_heading = normalize_angle(ship_state['heading'])
        heading_error = normalize_angle(attack_heading - current_heading)
        if heading_error > 180:
            heading_error -= 360

        relative_to_ship = normalize_angle(asteroid_angle - current_heading)
        if relative_to_ship > 180:
            relative_to_ship -= 360

        danger_sim = ctrl.ControlSystemSimulation(self.danger_ctrl)
        control_sim = ctrl.ControlSystemSimulation(self.control_ctrl)

        danger_sim.input['distance'] = closest_features['distance']
        danger_sim.input['relative_angle'] = relative_to_ship
        danger_sim.input['time_to_collision'] = closest_features['time_to_collision']
        danger_sim.compute()

        danger = danger_sim.output.get('danger', 100)

        control_sim.input['danger_input'] = danger
        control_sim.input['current_speed'] = ship_state['speed']
        control_sim.input['heading_error'] = heading_error
        control_sim.input['relative_angle'] = relative_to_ship
        control_sim.compute()

        thrust = control_sim.output.get('thrust', 0)
        turn_rate = control_sim.output.get('turn_rate', 0)

        self.eval_frames += 1
        return thrust, turn_rate, True, False

    def calculate_closest_asteroid_features(self, ship_state: Dict, game_state: Dict) -> Dict:
        ship_x = ship_state['position'][0]
        ship_y = ship_state['position'][1]
        asteroids = game_state['asteroids']
        closest_asteroid = None
        closest_distance = np.inf

        for asteroid in asteroids:
            asteroid_x = asteroid['position'][0]
            asteroid_y = asteroid['position'][1]
            center_distance = np.sqrt((ship_x - asteroid_x) ** 2 + (ship_y - asteroid_y) ** 2)
            edge_distance = center_distance - (ship_state['radius'] + asteroid['radius'])

            if edge_distance < closest_distance:
                closest_distance = edge_distance
                closest_asteroid = asteroid

        if closest_asteroid is None:
            raise RuntimeError("Closest asteroid not found")

        relative_angle = np.arctan2(closest_asteroid['position'][1] - ship_y, closest_asteroid['position'][0] - ship_x)
        asteroid_vx = closest_asteroid['velocity'][0]
        asteroid_vy = closest_asteroid['velocity'][1]
        ship_vx = ship_state['velocity'][0]
        ship_vy = ship_state['velocity'][1]

        relative_vx = asteroid_vx - ship_vx
        relative_vy = asteroid_vy - ship_vy

        time_to_collision = (closest_distance / np.sqrt(
            relative_vx ** 2 + relative_vy ** 2)) if relative_vx != 0 or relative_vy != 0 else np.inf

        return {
            "distance": closest_distance,
            "relative_angle": relative_angle,
            "time_to_collision": time_to_collision
        }

    def init_danger_fs(self):
        distance = ctrl.Antecedent(np.arange(0, 1000, 1), 'distance')
        relative_angle = ctrl.Antecedent(np.arange(-np.pi, np.pi, 0.01), 'relative_angle')
        time_to_collision = ctrl.Antecedent(np.arange(0, 60, 0.5), 'time_to_collision')
        danger = ctrl.Consequent(np.arange(0, 100, 1), 'danger')

        distance['critical'] = fuzz.trapmf(distance.universe, [0, 0, 30, 50])
        distance['close'] = fuzz.trapmf(distance.universe, [30, 50, 100, 150])
        distance['medium'] = fuzz.trapmf(distance.universe, [100, 150, 250, 300])
        distance['far'] = fuzz.trapmf(distance.universe, [250, 300, 400, 500])
        distance['safe'] = fuzz.trapmf(distance.universe, [400, 500, 1000, 1000])

        relative_angle['head_on'] = fuzz.trapmf(relative_angle.universe, [-0.2, -0.1, 0.1, 0.2])
        relative_angle['threatening_r'] = fuzz.trapmf(relative_angle.universe, [0.1, 0.2, 0.6, 0.8])
        relative_angle['threatening_l'] = fuzz.trapmf(relative_angle.universe, [-0.8, -0.6, -0.2, -0.1])
        relative_angle['oblique_r'] = fuzz.trapmf(relative_angle.universe, [0.6, 0.8, 1.2, 1.5])
        relative_angle['oblique_l'] = fuzz.trapmf(relative_angle.universe, [-1.5, -1.2, -0.8, -0.6])

        time_to_collision['imminent'] = fuzz.trapmf(time_to_collision.universe, [0, 0, 1, 2])
        time_to_collision['close'] = fuzz.trapmf(time_to_collision.universe, [1, 2, 3, 4])
        time_to_collision['medium'] = fuzz.trapmf(time_to_collision.universe, [3, 4, 6, 8])
        time_to_collision['far'] = fuzz.trapmf(time_to_collision.universe, [6, 8, 12, 15])
        time_to_collision['safe'] = fuzz.trapmf(time_to_collision.universe, [12, 15, 60, 60])

        danger['very_low'] = fuzz.trapmf(danger.universe, [0, 0, 10, 20])
        danger['low'] = fuzz.trapmf(danger.universe, [10, 20, 30, 40])
        danger['medium'] = fuzz.trapmf(danger.universe, [30, 40, 60, 70])
        danger['high'] = fuzz.trapmf(danger.universe, [60, 70, 80, 90])
        danger['very_high'] = fuzz.trapmf(danger.universe, [80, 90, 100, 100])

        danger_rules = [
            ctrl.Rule(time_to_collision['imminent'] & relative_angle['head_on'], danger['very_high']),
            ctrl.Rule(distance['critical'] & relative_angle['head_on'], danger['very_high']),
            ctrl.Rule(time_to_collision['imminent'] & distance['critical'], danger['very_high']),
        ]

        return ctrl.ControlSystem(danger_rules)

    def init_control_fs(self, ship_state: Dict):
        danger_input = ctrl.Antecedent(np.arange(0, 100, 1), 'danger_input')
        current_speed = ctrl.Antecedent(np.arange(0, ship_state['max_speed'], 1), 'current_speed')
        heading_error = ctrl.Antecedent(np.arange(-180, 180, 1), 'heading_error')
        relative_angle = ctrl.Antecedent(np.arange(-180, 180, 1), 'relative_angle')

        thrust = ctrl.Consequent(np.arange(ship_state['thrust_range'][0], ship_state['thrust_range'][1], 1), 'thrust')
        turn_rate = ctrl.Consequent(np.arange(ship_state['turn_rate_range'][0], ship_state['turn_rate_range'][1], 1),
                                    'turn_rate')

        thrust['reverse_full'] = fuzz.trapmf(thrust.universe, self.thrust_params)
        thrust['reverse_medium'] = fuzz.trapmf(thrust.universe, self.thrust_params)
        thrust['coast'] = fuzz.trapmf(thrust.universe, self.thrust_params)
        thrust['forward_medium'] = fuzz.trapmf(thrust.universe, self.thrust_params)
        thrust['forward_full'] = fuzz.trapmf(thrust.universe, self.thrust_params)

        turn_rate['sharp_left'] = fuzz.trapmf(turn_rate.universe, self.turn_rate_params)
        turn_rate['left'] = fuzz.trapmf(turn_rate.universe, self.turn_rate_params)
        turn_rate['straight'] = fuzz.trapmf(turn_rate.universe, self.turn_rate_params)
        turn_rate['right'] = fuzz.trapmf(turn_rate.universe, self.turn_rate_params)
        turn_rate['sharp_right'] = fuzz.trapmf(turn_rate.universe, self.turn_rate_params)

        control_rules = [
            ctrl.Rule(danger_input['very_high'] & relative_angle['front'], thrust['reverse_full']),
            ctrl.Rule(heading_error['large_negative'], turn_rate['sharp_left']),
        ]

        return ctrl.ControlSystem(control_rules)


if __name__ == "__main__":
    ga = EasyGA.GA()
    ga.gene_impl = lambda: [
        [np.random.uniform(-100, -50), np.random.uniform(-50, -20), np.random.uniform(-20, 20), np.random.uniform(20, 100)],
        [np.random.uniform(-150, -100), np.random.uniform(-100, -50), np.random.uniform(-50, 50), np.random.uniform(50, 150)]
    ]
    ga.chromosome_length = 2
    ga.population_size = 10
    ga.target_fitness_type = 'min'
    ga.generation_goal = 5
    ga.fitness_function_impl = fitness_function

    ga.evolve()

    best_chromosome = ga.best_chromosome
    thrust_params = best_chromosome[0].value
    turn_rate_params = best_chromosome[1].value

    controller = FuzzyThrustControllerReversed(thrust_params, turn_rate_params)

    scenario = Scenario(name="GA Final", num_asteroids=15, map_size=(1000, 800), ship_states=[{'position': (400, 400)}])
    game = KesslerGame(settings={'perf_tracker': True, 'graphics_type': None, 'realtime_multiplier': 1})
    game.run(scenario=scenario, controllers=[controller])
