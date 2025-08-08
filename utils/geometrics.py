
import numpy as np
import numba

def get_points_box_size(points):
    return np.sqrt(
        (np.max(points[:, 0]) - np.min(points[:, 0])) *
        (np.max(points[:, 1]) - np.min(points[:, 1]))
    )

def get_area_regular_polygon(n_sides, side_length):
    return 0.5 * n_sides * side_length * np.sqrt(3) * side_length / 2
    #return 0.25 * n_sides * side_length * side_length / np.tan(np.pi / n_sides)

def get_max_distance_reguar_polygon(n_points, side_length):
    if n_points <= 2:
        return side_length
    else:
        return side_length / np.sin(np.pi / n_points)