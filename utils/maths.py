import numba
import numpy as np


def get_unit_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])

def convert_angle_to_0_2pi(angle):
    return (angle + 2 * np.pi) % (2 * np.pi)

def convert_angle_to_minus_pi_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def norm_of_matrix_rows(matrix):
    return np.linalg.norm(matrix, axis=1)

def normalize_matrix_rows(matrix):
    return matrix / (np.expand_dims(norm_of_matrix_rows(matrix), axis=1) + 1e-8)

def line_circle_intersection(line_start, line_end, circle_center, circle_radius):
    d = (line_end - line_start).astype(np.float32)
    f = (line_start - circle_center).astype(np.float32)

    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - circle_radius**2

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        # no intersection
        return 0, np.array([0., 0.])
    else:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant)/(2*a)
        t2 = (-b + discriminant)/(2*a)
        if 0 <= t1 <= 1:
            return 1, line_start + t1 * d
        if 0 <= t2 <= 1:
            return 2, line_start + t2 * d
        return 0, np.array([0., 0.])

