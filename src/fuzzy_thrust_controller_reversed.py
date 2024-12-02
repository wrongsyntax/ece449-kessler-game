# fuzzy_thrust_controller_reversed.py

from typing import Dict, Tuple
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kesslergame import KesslerController

def normalize_angle(angle):
    """Normalize angle to [0, 360] range"""
    return angle % 360

class FuzzyThrustControllerReversed(KesslerController):
    def __init__(self, thrust_params=None, turn_rate_params=None):
        """
        Initialize the fuzzy controller with optional membership function parameters.
        :param thrust_params: Dictionary containing membership function boundaries for thrust.
        :param turn_rate_params: Dictionary containing membership function boundaries for turn_rate.
        """
        self.danger_ctrl = None
        self.control_ctrl = None
        self.eval_frames = 0

        # Default parameters if none provided
        if thrust_params is None:
            self.thrust_params = {
                'reverse_full': [ -100, -100, -100, -50],
                'reverse_medium': [ -70, -50, -20, -10],
                'coast': [ -10, -5, 5, 10],
                'forward_medium': [10, 20, 50, 70],
                'forward_full': [50, 100, 100, 100]
            }
        else:
            self.thrust_params = thrust_params

        if turn_rate_params is None:
            self.turn_rate_params = {
                'sharp_left': [ -150, -150, -150, -100],
                'left': [ -120, -120, -60, -60],
                'straight': [ -70, -5, 5, 70],
                'right': [60, 120, 120, 120],
                'sharp_right': [100, 150, 150, 150]
            }
        else:
            self.turn_rate_params = turn_rate_params

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # Set up the fuzzy control systems
        if self.danger_ctrl is None:
            self.danger_ctrl = self.init_danger_fs()
        if self.control_ctrl is None:
            self.control_ctrl = self.init_control_fs(ship_state)

        # Calculate inputs
        closest_features = self.calculate_closest_asteroid_features(ship_state, game_state)

        # Convert relative_angle from radians to degrees in [0, 360] range
        asteroid_angle = normalize_angle(np.degrees(closest_features['relative_angle']))

        # Calculate escape heading (opposite to asteroid)
        attack_heading = normalize_angle(asteroid_angle)

        # Get current heading in [0, 360] range
        current_heading = normalize_angle(ship_state['heading'])

        # Calculate heading error
        heading_error = normalize_angle(attack_heading - current_heading)
        # Convert to [-180, 180] range for the fuzzy system
        if heading_error > 180:
            heading_error -= 360

        # Calculate relative angle from ship's perspective
        relative_to_ship = normalize_angle(asteroid_angle - current_heading)
        # Convert to [-180, 180] range for the fuzzy system
        if relative_to_ship > 180:
            relative_to_ship -= 360

        # Set up the fuzzy control system simulators
        danger_sim = ctrl.ControlSystemSimulation(self.danger_ctrl)
        control_sim = ctrl.ControlSystemSimulation(self.control_ctrl)

        danger_sim.input['distance'] = closest_features['distance']
        danger_sim.input['relative_angle'] = relative_to_ship
        danger_sim.input['time_to_collision'] = closest_features['time_to_collision']

        danger_sim.compute()

        try:
            danger = danger_sim.output['danger']
        except KeyError:
            # Assume the worst if the danger is not defined
            danger = 100

        control_sim.input['danger_input'] = danger
        control_sim.input['current_speed'] = ship_state['speed']
        control_sim.input['heading_error'] = heading_error
        control_sim.input['relative_angle'] = relative_to_ship

        control_sim.compute()

        try:
            thrust = control_sim.output['thrust']
        except KeyError:
            thrust = 0

        try:
            turn_rate = control_sim.output['turn_rate']
        except KeyError:
            turn_rate = 0

        # Debug info
        print(f"Frame: {self.eval_frames}")
        print(f"Closest distance: {closest_features['distance']} px")
        print(f"Relative angle: {closest_features['relative_angle'] * 180 / np.pi}º")
        print(f"Time to collision: {closest_features['time_to_collision']} s")
        print(f"Danger: {danger}")
        print(f"Escape heading: {attack_heading}º")
        print(f"Curr heading: {ship_state['heading']}º")
        print(f"Heading error: {heading_error}º")
        print(f"Thrust: {thrust}")
        print(f"Turn rate: {turn_rate}")

        thrust = thrust
        turn_rate = turn_rate
        fire = True
        drop_mine = False
        self.eval_frames += 1
        return thrust, turn_rate, fire, drop_mine

    def calculate_closest_asteroid_features(self, ship_state: Dict, game_state: Dict) -> Dict:
        """
        Calculate the features of the closest asteroid to the ship. The features are:

        - distance: the distance to the asteroid (pixels)
        - relative_angle: the relative angle to the asteroid (radians)
        - time_to_collision: the time to collision with the asteroid (seconds)

        :param ship_state:
        :param game_state:
        :return:
        """

        # Get the ship position
        ship_x = ship_state['position'][0]
        ship_y = ship_state['position'][1]

        # Get the asteroids
        asteroids = game_state['asteroids']

        # Get the closest asteroid, so the features aren't calculated for all asteroids for no reason
        closest_asteroid = None
        closest_distance = np.inf

        for asteroid in asteroids:
            asteroid_x = asteroid['position'][0]
            asteroid_y = asteroid['position'][1]

            # Calculate center-to-center distance
            center_distance = np.sqrt((ship_x - asteroid_x) ** 2 + (ship_y - asteroid_y) ** 2)

            # Calculate actual distance by subtracting both radii
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

        speed = np.sqrt(relative_vx ** 2 + relative_vy ** 2)
        if speed != 0:
            time_to_collision = closest_distance / speed
        else:
            time_to_collision = np.inf

        return {
            "distance": closest_distance,
            "relative_angle": relative_angle,
            "time_to_collision": time_to_collision
        }

    def init_danger_fs(self):
        # Antecedent variables
        distance = ctrl.Antecedent(np.arange(0, 1000, 1), 'distance')
        relative_angle = ctrl.Antecedent(np.arange(-np.pi, np.pi, 0.01), 'relative_angle')
        time_to_collision = ctrl.Antecedent(np.arange(0, 60, 0.5), 'time_to_collision')

        # Consequent variables
        danger = ctrl.Consequent(np.arange(0, 100, 1), 'danger')

        distance['critical'] = fuzz.trapmf(distance.universe, [0, 0, 30, 50])
        distance['close'] = fuzz.trapmf(distance.universe, [30, 50, 100, 150])
        distance['medium'] = fuzz.trapmf(distance.universe, [100, 150, 250, 300])
        distance['far'] = fuzz.trapmf(distance.universe, [250, 300, 400, 500])
        distance['safe'] = fuzz.trapmf(distance.universe, [400, 500, 1000, 1000])

        # Relative angle membership functions
        relative_angle['head_on'] = fuzz.trapmf(relative_angle.universe, [-0.2, -0.1, 0.1, 0.2])
        relative_angle['threatening_r'] = fuzz.trapmf(relative_angle.universe, [0.1, 0.2, 0.6, 0.8])
        relative_angle['threatening_l'] = fuzz.trapmf(relative_angle.universe, [-0.8, -0.6, -0.2, -0.1])
        relative_angle['oblique_r'] = fuzz.trapmf(relative_angle.universe, [0.6, 0.8, 1.2, 1.5])
        relative_angle['oblique_l'] = fuzz.trapmf(relative_angle.universe, [-1.5, -1.2, -0.8, -0.6])
        relative_angle['safe_r'] = fuzz.trapmf(relative_angle.universe, [1.2, 1.5, np.pi, np.pi])
        relative_angle['safe_l'] = fuzz.trapmf(relative_angle.universe, [-np.pi, -np.pi, -1.5, -1.2])

        # Time to collision membership functions
        time_to_collision['imminent'] = fuzz.trapmf(time_to_collision.universe, [0, 0, 1, 2])
        time_to_collision['close'] = fuzz.trapmf(time_to_collision.universe, [1, 2, 3, 4])
        time_to_collision['medium'] = fuzz.trapmf(time_to_collision.universe, [3, 4, 6, 8])
        time_to_collision['far'] = fuzz.trapmf(time_to_collision.universe, [6, 8, 12, 15])
        time_to_collision['safe'] = fuzz.trapmf(time_to_collision.universe, [12, 15, 60, 60])

        # Danger membership functions
        danger['very_low'] = fuzz.trapmf(danger.universe, [0, 0, 10, 20])
        danger['low'] = fuzz.trapmf(danger.universe, [10, 20, 30, 40])
        danger['medium'] = fuzz.trapmf(danger.universe, [30, 40, 60, 70])
        danger['high'] = fuzz.trapmf(danger.universe, [60, 70, 80, 90])
        danger['very_high'] = fuzz.trapmf(danger.universe, [80, 90, 100, 100])

        # Add rules
        danger_rules = [
            # Default rule - if no other rules match
            ctrl.Rule(~(time_to_collision['imminent'] | time_to_collision['close'] | time_to_collision['medium']),
                      danger['very_low']),

            # Critical situations
            ctrl.Rule(time_to_collision['imminent'] & relative_angle['head_on'], danger['very_high']),
            ctrl.Rule(distance['critical'] & relative_angle['head_on'], danger['very_high']),
            ctrl.Rule(time_to_collision['imminent'] & distance['critical'], danger['very_high']),

            # High danger situations
            ctrl.Rule(time_to_collision['close'] & relative_angle['threatening_l'], danger['high']),
            ctrl.Rule(time_to_collision['close'] & relative_angle['threatening_r'], danger['high']),
            ctrl.Rule(distance['close'] & relative_angle['head_on'], danger['high']),
            ctrl.Rule(distance['critical'] & (relative_angle['threatening_l'] | relative_angle['threatening_r']),
                      danger['high']),

            # Medium danger situations
            ctrl.Rule(time_to_collision['medium'] & relative_angle['head_on'], danger['medium']),
            ctrl.Rule(distance['medium'] & (relative_angle['threatening_l'] | relative_angle['threatening_r']),
                      danger['medium']),
            ctrl.Rule(distance['close'] & (relative_angle['oblique_l'] | relative_angle['oblique_r']),
                      danger['medium']),

            # Low danger situations
            ctrl.Rule(time_to_collision['far'], danger['low']),
            ctrl.Rule(distance['far'], danger['low']),
            ctrl.Rule((relative_angle['safe_l'] | relative_angle['safe_r']) & distance['medium'],
                      danger['low']),

            # Very low danger situations
            ctrl.Rule(time_to_collision['safe'] & distance['safe'], danger['very_low']),
            ctrl.Rule((relative_angle['safe_l'] | relative_angle['safe_r']) & distance['safe'],
                      danger['very_low'])
        ]

        return ctrl.ControlSystem(danger_rules)

    def init_control_fs(self, ship_state: Dict):
        # Antecedent variables
        danger_input = ctrl.Antecedent(np.arange(0, 100, 1), 'danger_input')
        current_speed = ctrl.Antecedent(np.arange(0, ship_state['max_speed'], 1), 'current_speed')
        heading_error = ctrl.Antecedent(np.arange(-180, 180, 1),'heading_error')
        relative_angle = ctrl.Antecedent(np.arange(-180, 180, 1), 'relative_angle')

        # Consequent variables
        thrust = ctrl.Consequent(np.arange(ship_state['thrust_range'][0], ship_state['thrust_range'][1], 1), 'thrust')
        turn_rate = ctrl.Consequent(np.arange(ship_state['turn_rate_range'][0], ship_state['turn_rate_range'][1], 1),
                                    'turn_rate')

        # Membership functions
        danger_input.automf(5, names=['very_low', 'low', 'medium', 'high', 'very_high'])
        current_speed.automf(4, names=['stopped', 'slow', 'medium', 'fast'])

        heading_error['large_negative'] = fuzz.trapmf(heading_error.universe, [-180, -180, -60, -30])
        heading_error['small_negative'] = fuzz.trapmf(heading_error.universe, [-45, -30, -10, -5])
        heading_error['zero'] = fuzz.trapmf(heading_error.universe, [-5, -2, 2, 5])
        heading_error['small_positive'] = fuzz.trapmf(heading_error.universe, [5, 10, 30, 45])
        heading_error['large_positive'] = fuzz.trapmf(heading_error.universe, [30, 60, 180, 180])

        # Relative angle memberships (where the asteroid actually is)
        relative_angle['front'] = fuzz.trapmf(relative_angle.universe, [-45, -20, 20, 45])
        relative_angle['front_left'] = fuzz.trapmf(relative_angle.universe, [-90, -70, -40, -20])
        relative_angle['front_right'] = fuzz.trapmf(relative_angle.universe, [20, 40, 70, 90])
        relative_angle['back_left'] = fuzz.trapmf(relative_angle.universe, [-180, -180, -110, -90])
        relative_angle['back_right'] = fuzz.trapmf(relative_angle.universe, [90, 110, 180, 180])

        # Define thrust membership functions using parameters
        thrust['reverse_full'] = fuzz.trapmf(thrust.universe, self.thrust_params['reverse_full'])
        thrust['reverse_medium'] = fuzz.trapmf(thrust.universe, self.thrust_params['reverse_medium'])
        thrust['coast'] = fuzz.trapmf(thrust.universe, self.thrust_params['coast'])
        thrust['forward_medium'] = fuzz.trapmf(thrust.universe, self.thrust_params['forward_medium'])
        thrust['forward_full'] = fuzz.trapmf(thrust.universe, self.thrust_params['forward_full'])

        # Define turn_rate membership functions using parameters
        turn_rate['sharp_left'] = fuzz.trapmf(turn_rate.universe, self.turn_rate_params['sharp_left'])
        turn_rate['left'] = fuzz.trapmf(turn_rate.universe, self.turn_rate_params['left'])
        turn_rate['straight'] = fuzz.trapmf(turn_rate.universe, self.turn_rate_params['straight'])
        turn_rate['right'] = fuzz.trapmf(turn_rate.universe, self.turn_rate_params['right'])
        turn_rate['sharp_right'] = fuzz.trapmf(turn_rate.universe, self.turn_rate_params['sharp_right'])

        # Rules
        control_rules = [
            # Emergency avoidance rules
            # If asteroid is in front, and we're not pointed away, reverse
            ctrl.Rule(
                danger_input['very_high'] & relative_angle['front'] & ~heading_error['large_negative'] & ~heading_error['large_positive'],
                thrust['reverse_full']),

            # If asteroid is in front-left or front-right and we're not pointed away, reverse
            ctrl.Rule(danger_input['very_high'] & (relative_angle['front_left'] | relative_angle['front_right']) &
                      ~heading_error['large_negative'] & ~heading_error['large_positive'],
                      thrust['reverse_medium']),

            # If asteroid is behind (back_left or back_right) and we're pointed away, go forward
            ctrl.Rule(danger_input['very_high'] & (relative_angle['back_left'] | relative_angle['back_right']) &
                      heading_error['zero'],
                      thrust['forward_full']),

            # High danger responses
            ctrl.Rule(danger_input['high'] & relative_angle['front'] & heading_error['zero'],
                      thrust['reverse_medium']),

            ctrl.Rule(
                danger_input['high'] & (relative_angle['back_left'] | relative_angle['back_right']) & heading_error['zero'],
                thrust['forward_medium']),

            # Medium danger - more conservative
            ctrl.Rule(danger_input['medium'] & current_speed['fast'],
                      thrust['coast']),

            ctrl.Rule(danger_input['medium'] & current_speed['stopped'] & relative_angle['front'],
                      thrust['reverse_medium']),

            # When well-aligned and not too close, move forward
            ctrl.Rule(heading_error['zero'] & ~danger_input['very_high'],
                      thrust['forward_medium']),

            # When perfectly aligned and at safe distance, full thrust
            ctrl.Rule(heading_error['zero'] & danger_input['low'],
                      thrust['forward_full']),

            # When turning but at safe distance, maintain some forward momentum
            ctrl.Rule((heading_error['small_negative'] | heading_error['small_positive']) & danger_input['low'],
                      thrust['forward_medium']),

            # When stopped and target in sight, start moving
            ctrl.Rule(current_speed['stopped'] & heading_error['zero'] & ~danger_input['very_high'],
                      thrust['forward_medium']),

            # Only reverse if in extreme danger and very close
            ctrl.Rule(danger_input['very_high'] & relative_angle['front'] & current_speed['fast'],
                      thrust['reverse_full']),

            # Otherwise try to turn and maintain forward momentum
            ctrl.Rule(danger_input['high'] & ~relative_angle['front'],
                      thrust['forward_medium']),

            # When danger is moderate, maintain speed
            ctrl.Rule(danger_input['medium'],
                      thrust['coast']),

            # Low danger - tend towards stopping
            ctrl.Rule(danger_input['low'] & current_speed['fast'],
                      thrust['coast']),

            # Turning rules
            ctrl.Rule(heading_error['large_negative'], turn_rate['sharp_left']),
            ctrl.Rule(heading_error['small_negative'], turn_rate['left']),
            ctrl.Rule(heading_error['zero'], turn_rate['straight']),
            ctrl.Rule(heading_error['small_positive'], turn_rate['right']),
            ctrl.Rule(heading_error['large_positive'], turn_rate['sharp_right'])
        ]

        return ctrl.ControlSystem(control_rules)

    @property
    def name(self) -> str:
        return "Fuzzy Thrust Controller"
