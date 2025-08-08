from . import obstacle_avoiding, path_steering

algorithm_dir = {
    "OBSTACLE_AVOIDING": obstacle_avoiding.compute_obstacle_avoiding_action,
    "PATH_STEERING": path_steering.compute_path_steering_action,
}


def get_algorithm_names():
    return list(algorithm_dir.keys())


def get_algorithm(algorithm_name):
    return algorithm_dir[algorithm_name]
