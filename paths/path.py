import pickle
import numpy as np
import json

from scipy.interpolate import CubicSpline, CubicHermiteSpline
from scipy.stats import norm

from shapelysmooth import chaikin_smooth, catmull_rom_smooth
from shapely import LineString

from paths import OUTPUT_FOLDER, draw_linear_path, draw_smooth_path
from utils.geometrics import get_max_distance_reguar_polygon

# Help functions


def generate_normal_random_angle(prev_angle, angle_normal_distribution_params):
    angle = np.random.normal(*angle_normal_distribution_params)
    angle_sign = np.random.choice([-1, 1])
    angle = prev_angle + angle_sign * angle
    angle = max(-np.pi / 2 + 0.01, min(np.pi / 2 - 0.01, angle))
    return angle


def generate_uniform_random_angle(prev_angle, angle_threshold):
    min_angle = max(-np.pi / 2, prev_angle - angle_threshold)
    max_angle = min(np.pi / 2, prev_angle + angle_threshold)
    angle = np.random.uniform(min_angle, max_angle)
    return angle


def generate_random_positive_angle():
    return np.random.uniform(0, np.pi / 2)


def generate_random_angle(prev_angle):
    if prev_angle > 0:
        angle = np.random.uniform(-np.pi / 2, 0)
    else:
        angle = np.random.uniform(0, np.pi / 2)
    return angle


def generate_random_angle_cutoff(prev_angle):
    if prev_angle > 0:
        angle = np.random.uniform(-np.pi / 2, prev_angle)
    else:
        angle = np.random.uniform(prev_angle, np.pi / 2)
    return angle


def clean_up_margin(is_bot, x_space, y_space):
    indexes_to_remove = []
    for i in range(1, len(x_space)):
        current_x = x_space[i]
        previous_x = x_space[i - 1]

        if current_x < previous_x:
            indexes = np.where(
                np.logical_and(x_space >= current_x, x_space < previous_x)
            )[0]
        else:
            indexes = np.where(
                np.logical_and(x_space > previous_x, x_space <= current_x)
            )[0]

        if len(indexes) == 0:
            pass
        else:
            if is_bot:
                local_opt = np.min(y_space[indexes])
            else:
                local_opt = np.max(y_space[indexes])
            for index in indexes:
                if is_bot:
                    if y_space[index] > local_opt:
                        indexes_to_remove.append(index)
                else:
                    if y_space[index] < local_opt:
                        indexes_to_remove.append(index)
    return indexes_to_remove


def clean_up_margin_ends(
    x_space, y_space, bot_x_space, bot_y_space, top_x_space, top_y_space
):
    indexes_to_remove = []
    for i in range(x_space.shape[0]):
        x = x_space[i]
        y = y_space[i]
        if (y > np.interp(x, bot_x_space, bot_y_space)) and (
            y < np.interp(x, top_x_space, top_y_space)
        ):
            indexes_to_remove.append(i)

    return indexes_to_remove


# Class


class Path:
    def __init__(
        self, type=None, nodes=[], spline=None, x_space_num=100, y_space_num=100
    ):
        self.type = type
        self.nodes = nodes
        self.spline = spline

        self.x_space = None
        self.x_range = 0.0
        self.x_space_num = x_space_num

        self.y_space = None
        self.y_range = 0.0
        self.y_space_num = y_space_num
        self.y_values = None

        self.has_margin = False
        self.top_margin_x_space = None
        self.top_margin_y_space = None
        self.bot_margin_x_space = None
        self.bot_margin_y_space = None

        self.top_safety_margin_x_space = None
        self.top_safety_margin_y_space = None
        self.bot_safety_margin_x_space = None
        self.bot_safety_margin_y_space = None

        self.derivatives = None
        self.margin_values = None

        self.characteristics = {}

        self.is_repositioned = False

        self.end_point_radius = None
        self.end_point = None

    def reposition(self, x_offset, y_offset):
        self.nodes[:, 0] += x_offset
        self.nodes[:, 1] += y_offset

        self.x_space += x_offset
        self.y_values += y_offset

        self.top_margin_x_space += x_offset
        self.top_margin_y_space += y_offset
        self.bot_margin_x_space += x_offset
        self.bot_margin_y_space += y_offset

        self.top_safety_margin_x_space += x_offset
        self.top_safety_margin_y_space += y_offset
        self.bot_safety_margin_x_space += x_offset
        self.bot_safety_margin_y_space += y_offset

        self.recompute_linear_variables()
        self.is_repositioned = True

    def recompute_linear_variables(self):
        self.x_range = (np.min(self.nodes[:, 0]), np.max(self.nodes[:, 0]))
        self.x_space_num = len(self.x_space)

        self.end_point = (self.x_space[-1], self.y_values[-1])

    def set_linear_spaces(self):
        self.x_space = np.linspace(
            min(self.nodes[:, 0]), max(self.nodes[:, 0]), num=self.x_space_num
        )
        self.y_values = np.interp(self.x_space, self.nodes[:, 0], self.nodes[:, 1])

        array = np.array([self.x_space, self.y_values]).T
        array_smoothed = np.array(chaikin_smooth(list(map(tuple, array)), iters=5))

        self.x_space = array_smoothed[:, 0]
        self.y_values = array_smoothed[:, 1]

        self.y_range = (np.min(self.y_values), np.max(self.y_values))

    def set_smooth_spaces(self):
        self.y_values = self.spline(self.x_space)
        self.y_space = np.linspace(
            min(self.y_values), max(self.y_values), num=self.y_space_num
        )
        self.y_range = (np.min(self.y_values), np.max(self.y_values))
    
    def generate_zero_turns(
        self,
        total_length,
    ):
        # Create nodes
        nodes = [np.array([0.0, 0.0])]
        
        nodes.append(nodes[0] + np.array([total_length, 0]))

        self.type = "linear"
        self.nodes = np.array(nodes)

        self.set_linear_spaces()
        self.recompute_linear_variables()
    
    def generate_one_turn(
        self,
        turn_degree,
        margin_dx, 
        gaussian_scaling,
        min_margin,
        max_margin,
        safety_margin_multiplier,
    ):
        # Create nodes
        nodes = [np.array([0.0, 0.0])]
        
        seg_len = 1000
        nodes.append(nodes[0] + np.array([seg_len, 0]))

        seg2_len = 2000
        angle = np.deg2rad(turn_degree)
        dx = seg2_len * np.cos(angle)
        dy = seg2_len * np.sin(angle)
        nodes.append(nodes[1] + np.array([dx, dy]))

        self.type = "linear"
        self.nodes = np.array(nodes)

        self.set_linear_spaces()
        self.recompute_linear_variables()

        # Create margin pdf
        gaussian_distributions = []

        class Distribution:
            def pdf(self, x):
                if x < seg_len:
                    return max_margin
                else:
                    if x >= seg_len + margin_dx:
                        return 0
                    else:
                        return max_margin + (min_margin - max_margin) / margin_dx * (x - seg_len) - min_margin

        gaussian_distributions.append(Distribution())
        
        # Generate margin
        self._generate_margin(
            gaussian_distributions,
            n_mixtures=1,
            gaussian_scaling=gaussian_scaling,
            min_margin=min_margin,
            max_margin=max_margin,
            cte_margin=None,
            safety_margin_multiplier=safety_margin_multiplier,
        )

    def generate_new_linear(
        self,
        total_length=5000,
        segment_normal_distrubtion_params=(200, 50),
        angle_normal_distribution_params=(np.pi / 3, np.pi / 6),
        angle_distribution="random",
    ):
        nodes = [np.array([0.0, 0.0])]
        len = 0
        i = 0
        prev_angle = 0
        while len < total_length:
            seg_len = np.random.normal(*segment_normal_distrubtion_params)
            if i > 0:
                if angle_distribution == "random":
                    angle = generate_random_angle(prev_angle)
                elif angle_distribution == "random_positive":
                    angle = generate_random_positive_angle()
                else:
                    raise ValueError(
                        "Unknown angle distribution: {}".format(angle_distribution)
                    )
                prev_angle = angle
            else:
                angle = 0
            dx = seg_len * np.cos(angle)
            dy = seg_len * np.sin(angle)
            nodes.append(nodes[i] + np.array([dx, dy]))
            i += 1
            len += seg_len

        self.type = "linear"
        self.nodes = np.array(nodes)

        self.set_linear_spaces()
        self.recompute_linear_variables()

    def generate_new_smooth(
        self,
        smoothing="cubic",
        total_length=5000,
        segment_normal_distrubtion_params=(200, 50),
        angle_normal_distribution_params=(0, np.pi / 6),
        angle_distribution="random",
    ):
        self.generate_new_linear(
            total_length=total_length,
            segment_normal_distrubtion_params=segment_normal_distrubtion_params,
            angle_normal_distribution_params=angle_normal_distribution_params,
            angle_distribution=angle_distribution,
        )

        if smoothing == "cubic":
            cs = CubicSpline(self.nodes[:, 0], self.nodes[:, 1])
        elif smoothing == "cubic_hermite":
            cs = CubicHermiteSpline(
                self.nodes[:, 0], self.nodes[:, 1], np.zeros(len(self.nodes[:, 0]))
            )
        else:
            raise ValueError("Unknown smoothing method: {}".format(smoothing))

        self.type = smoothing
        self.spline = cs

        self.set_smooth_spaces()
        self.set_smooth_characteristics()

    def generate_margin(
        self,
        n_mixtures,
        gaussian_scaling,
        min_margin,
        max_margin,
        min_margin_multiplier,
        max_margin_multiplier,
        cte_margin=None,
        safety_margin=0.0,
        safety_margin_multiplier=1.0,
    ):
        self.has_margin = True

        # Recompute parameters
        min_margin = min_margin_multiplier * min_margin
        max_margin = min(max_margin_multiplier * min_margin, max_margin)
        # safety_margin = safety_margin + safety_margin_multiplier * min_margin

        # Create Gaussian mixtures
        mixture_delimiter = (self.x_range[1] - self.x_range[0]) / n_mixtures
        gaussian_distributions = []
        for i in range(n_mixtures):
            # norm_i = norm(
            #     i * mixture_delimiter + 0.5 * mixture_delimiter, mixture_delimiter / 4
            # )
            norm_i = norm(
                i * mixture_delimiter + 0.5 * mixture_delimiter, mixture_delimiter / 10
            )
            gaussian_distributions.append(norm_i)
        
        # Generate margin
        self._generate_margin(
            gaussian_distributions,
            n_mixtures,
            gaussian_scaling,
            min_margin,
            max_margin,
            cte_margin,
            safety_margin_multiplier,
        )
    
    def _generate_margin(
        self,
        gaussian_distributions,
        n_mixtures,
        gaussian_scaling,
        min_margin,
        max_margin,
        cte_margin,
        safety_margin_multiplier,
    ):
        # Compute margins
        top_margin_x_array = np.zeros((self.x_space_num, 2))
        bot_margin_x_array = np.zeros((self.x_space_num, 2))

        top_safety_margin_x_array = np.zeros((self.x_space_num, 2))
        bot_safety_margin_x_array = np.zeros((self.x_space_num, 2))

        derivatives = []
        margin_values = []
        for i in range(self.x_space_num):
            if i < self.x_space_num - 1:
                if cte_margin is not None:
                    margin_value = cte_margin
                else:
                    margin_value = min_margin
                    for j in range(n_mixtures):
                        margin_value_j = (
                            min_margin
                            + gaussian_scaling
                            * gaussian_distributions[j].pdf(self.x_space[i])
                        )
                        margin_value = max(margin_value, margin_value_j)
                    margin_value = max(min_margin, min(max_margin, margin_value))
                margin_values.append(margin_value)

                derivative = (self.y_values[i + 1] - self.y_values[i]) / (
                    self.x_space[i + 1] - self.x_space[i]
                )
                derivatives.append(derivative)
                derivative_angle = np.arctan(derivative)
                margin_direction = np.array(
                    [
                        np.cos(derivative_angle + np.pi / 2),
                        np.sin(derivative_angle + np.pi / 2),
                    ]
                )

                margin_point = margin_value * margin_direction

                top_margin_x_array[i] = margin_point
                bot_margin_x_array[i] = margin_value * np.array(
                    [
                        np.cos(derivative_angle - np.pi / 2),
                        np.sin(derivative_angle - np.pi / 2),
                    ]
                )

                # safety_margin_value = margin_value - safety_margin
                safety_margin_value = (1 - safety_margin_multiplier) * margin_value
                safety_margin_point = safety_margin_value * margin_direction

                top_safety_margin_x_array[i] = safety_margin_point
                bot_safety_margin_x_array[i] = safety_margin_value * np.array(
                    [
                        np.cos(derivative_angle - np.pi / 2),
                        np.sin(derivative_angle - np.pi / 2),
                    ]
                )
            else:
                top_margin_x_array[i] = top_margin_x_array[i - 1]
                bot_margin_x_array[i] = bot_margin_x_array[i - 1]
                top_safety_margin_x_array[i] = top_safety_margin_x_array[i - 1]
                bot_safety_margin_x_array[i] = bot_safety_margin_x_array[i - 1]
        self.derivatives = np.array(derivatives)
        self.margin_values = np.array(margin_values)

        self.end_point_radius = margin_values[-1]

        # Actual
        top_margin_x_space = self.x_space + top_margin_x_array[:, 0]
        top_margin_y_space = self.y_values + top_margin_x_array[:, 1]

        bot_margin_x_space = self.x_space + bot_margin_x_array[:, 0]
        bot_margin_y_space = self.y_values + bot_margin_x_array[:, 1]

        indexes_to_remove = clean_up_margin(
            False, top_margin_x_space, top_margin_y_space
        )
        self.top_margin_x_space = np.delete(top_margin_x_space, indexes_to_remove)
        self.top_margin_y_space = np.delete(top_margin_y_space, indexes_to_remove)

        indexes_to_remove = clean_up_margin(
            True, bot_margin_x_space, bot_margin_y_space
        )
        self.bot_margin_x_space = np.delete(bot_margin_x_space, indexes_to_remove)
        self.bot_margin_y_space = np.delete(bot_margin_y_space, indexes_to_remove)

        indexes_to_remove = clean_up_margin_ends(
            self.bot_margin_x_space,
            self.bot_margin_y_space,
            bot_margin_x_space,
            bot_margin_y_space,
            top_margin_x_space,
            top_margin_y_space,
        )
        self.bot_margin_x_space = np.delete(self.bot_margin_x_space, indexes_to_remove)
        self.bot_margin_y_space = np.delete(self.bot_margin_y_space, indexes_to_remove)

        indexes_to_remove = clean_up_margin_ends(
            self.top_margin_x_space,
            self.top_margin_y_space,
            bot_margin_x_space,
            bot_margin_y_space,
            top_margin_x_space,
            top_margin_y_space,
        )
        self.top_margin_x_space = np.delete(self.top_margin_x_space, indexes_to_remove)
        self.top_margin_y_space = np.delete(self.top_margin_y_space, indexes_to_remove)

        # Safety
        top_safety_margin_x_space = self.x_space + top_safety_margin_x_array[:, 0]
        top_safety_margin_y_space = self.y_values + top_safety_margin_x_array[:, 1]

        bot_safety_margin_x_space = self.x_space + bot_safety_margin_x_array[:, 0]
        bot_safety_margin_y_space = self.y_values + bot_safety_margin_x_array[:, 1]

        indexes_to_remove = clean_up_margin(
            False, top_safety_margin_x_space, top_safety_margin_y_space
        )
        self.top_safety_margin_x_space = np.delete(
            top_safety_margin_x_space, indexes_to_remove
        )
        self.top_safety_margin_y_space = np.delete(
            top_safety_margin_y_space, indexes_to_remove
        )

        indexes_to_remove = clean_up_margin(
            True, bot_safety_margin_x_space, bot_safety_margin_y_space
        )
        self.bot_safety_margin_x_space = np.delete(
            bot_safety_margin_x_space, indexes_to_remove
        )
        self.bot_safety_margin_y_space = np.delete(
            bot_safety_margin_y_space, indexes_to_remove
        )

        self.set_characteristics()

    def is_in_margin(self, x, y):
        return (
            y > np.interp(x, self.bot_margin_x_space, self.bot_margin_y_space)
        ) and (y < np.interp(x, self.top_margin_x_space, self.top_margin_y_space))

    def margin_error(self, x, y):
        if y > np.interp(x, self.bot_margin_x_space, self.bot_margin_y_space):
            return np.abs(
                y - np.interp(x, self.top_margin_x_space, self.top_margin_y_space)
            )
        if y < np.interp(x, self.top_margin_x_space, self.top_margin_y_space):
            return np.abs(
                y - np.interp(x, self.bot_margin_x_space, self.bot_margin_y_space)
            )
        return 0
    
    def has_finished(self, poses, end_point_distance=200):
        return np.any(poses[:, 0] >= self.end_point[0] - end_point_distance)
    
    def is_at_finish(self, poses, end_point_radius=None):
        if end_point_radius is None:
            end_point_radius = self.end_point_radius
        return np.any((poses[:, 0] - self.end_point[0])**2 + (poses[:, 1] - self.end_point[1])**2 < end_point_radius**2)
    
    # def get_derivative(self, x, y):
    #     distances = np.sqrt((x - self.x_space) ** 2 + (y - self.y_values) ** 2)
    #     index = np.argmin(distances)
    #     derivative = self.derivatives[index]
    #     return derivative

    def get_gradient(self, x, y):
        is_in_safety_margin = False

        distances = np.sqrt((x - self.x_space) ** 2 + (y - self.y_values) ** 2)
        index = np.argmin(distances)
        nearest_point = np.array([self.x_space[index], self.y_values[index]])

        if (y > np.interp(x, self.bot_margin_x_space, self.bot_margin_y_space)) and (
            y < np.interp(x, self.top_margin_x_space, self.top_margin_y_space)
        ):
            if index == self.x_space_num - 1:
                index = self.x_space_num - 2
            derivative = self.derivatives[index]

            derivative_angle = np.arctan(derivative)
            gradient = np.array(
                [
                    np.cos(derivative_angle),
                    np.sin(derivative_angle),
                ]
            )

            if (
                y
                > np.interp(
                    x, self.bot_safety_margin_x_space, self.bot_safety_margin_y_space
                )
            ) and (
                y
                < np.interp(
                    x, self.top_safety_margin_x_space, self.top_safety_margin_y_space
                )
            ):
                is_in_safety_margin = True
            else:
                path_direction = nearest_point - np.array([x, y])
                path_direction = path_direction / np.linalg.norm(path_direction)

                # gradient = 0.5 * gradient + 0.5 * path_direction
                gradient = path_direction
        else:
            gradient = nearest_point - np.array([x, y])
            gradient = gradient / np.linalg.norm(gradient)
        return gradient, nearest_point, is_in_safety_margin

    def get_potential(self, x, y, k=1):
        min_margin_potential = 0.0
        max_margin_potential = -2000.0

        if (
            y
            > np.interp(
                x, self.bot_safety_margin_x_space, self.bot_safety_margin_y_space
            )
        ) and (
            y
            < np.interp(
                x, self.top_safety_margin_x_space, self.top_safety_margin_y_space
            )
        ):
            x_range = self.x_range[1] - self.x_range[0]

            potential = (
                max_margin_potential * ((x - self.x_range[0]) / x_range)
                + min_margin_potential
            )
        else:
            distances = np.sqrt((x - self.x_space) ** 2 + (y - self.y_values) ** 2)
            index = np.argmin(distances)

            distance = distances[index]
            nearest_point = np.array([self.x_space[index], self.y_values[index]])

            potential = distance
        return potential

    def set_characteristics(self):
        if self.type == "linear":
            self.set_linear_characteristics()
        elif self.type in ["cubic", "cubic_hermite"]:
            self.set_smooth_characteristics()
        else:
            raise ValueError("Unknown path type: {}".format(self.type))

    def set_linear_characteristics(self):
        distances_local_optimum = []
        i = 0
        while i < self.nodes.shape[0] - 1:
            distance = np.linalg.norm(self.nodes[i + 1] - self.nodes[i])
            distances_local_optimum.append(distance)
            i += 1

        self.characteristics.update(
            {
                "num_local_optimum": self.nodes.shape[0],
                "x_local_optimum": list(self.nodes[:, 0]),
                "y_local_optimum": list(self.nodes[:, 1]),
                "distances_local_optimum": distances_local_optimum,
            }
        )

    def set_smooth_characteristics(self):
        num_local_optimum = 0
        x_local_optimum = []
        i = 1
        while i < len(self.derivatives):
            if (self.derivatives[i - 1] < 0 and self.derivatives[i] > 0) or (
                self.derivatives[i - 1] > 0 and self.derivatives[i] < 0
            ):
                num_local_optimum += 1
                x_local_optimum.append((self.x_space[i - 1] + self.x_space[i]) / 2)
            i += 1
        i = 1

        y_local_optimum = list(self.spline(x_local_optimum))

        distances_local_optimum = []
        while i < len(x_local_optimum):
            opt_1 = np.array([x_local_optimum[i - 1], y_local_optimum[i - 1]])
            opt_2 = np.array([x_local_optimum[i], y_local_optimum[i]])
            distance = np.linalg.norm(opt_1 - opt_2)
            distances_local_optimum.append(distance)
            i += 1

        self.characteristics.update(
            {
                "num_local_optimum": num_local_optimum,
                "x_local_optimum": x_local_optimum,
                "y_local_optimum": y_local_optimum,
                "distances_local_optimum": distances_local_optimum,
            }
        )

    def get_characteristics(self):
        num_local_optimum = self.characteristics["num_local_optimum"]
        x_local_optimum = self.characteristics["x_local_optimum"]
        y_local_optimum = self.characteristics["y_local_optimum"]
        distances_local_optimum = self.characteristics["distances_local_optimum"]

        return (
            num_local_optimum,
            x_local_optimum,
            y_local_optimum,
            distances_local_optimum,
        )

    def draw(self, out_filename="path_new", out_folder=OUTPUT_FOLDER):
        if self.type == "linear":
            draw_linear_path(
                path=self,
                show_directon=True,
                out_filename=out_filename,
                out_folder=out_folder,
            )
        elif self.type in ["cubic", "cubic_hermite"]:
            draw_smooth_path(
                path=self, out_filename=out_filename, out_folder=out_folder
            )
        else:
            raise ValueError("Unknown path type: {}".format(self.type))

    def save(self, out_filename="path_new", out_folder=None):
        self.save_characteristics_json(out_filename=out_filename, out_folder=out_folder)
        self.save_path_pickle(out_filename=out_filename, out_folder=out_folder)

    def save_characteristics_json(self, out_filename="path_new", out_folder=None):
        if out_folder is None:
            out_folder = OUTPUT_FOLDER
        with open(f"{out_folder}{out_filename}_char.json", "w") as f:
            json.dump(self.characteristics, f)

    def save_path_pickle(self, out_filename="path_new", out_folder=None):
        if out_folder is None:
            out_folder = OUTPUT_FOLDER
        with open(f"{out_folder}{out_filename}.pkl", "wb") as f:
            pickle.dump(self, f)

    def print(self):
        nodes_print = []
        for node in self.nodes:
            nodes_print.append(node.tolist())
        print(nodes_print)
