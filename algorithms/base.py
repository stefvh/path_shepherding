import numpy as np

from utils import maths


def handle_collision_avoidance_hard(
    distances_squared, focal_pose, other_poses, avoidance_boundary
):
    indexes = np.logical_and(
        0.0 < distances_squared, distances_squared <= avoidance_boundary**2
    )
    b_collision = np.any(indexes)
    d_theta, d_v = 0, 0
    if b_collision:
        q = -np.sum(
            maths.normalize_matrix_rows(other_poses[indexes, :2] - focal_pose[:2]),
            axis=0,
        )
        d_theta = np.arctan2(q[1], q[0]) % (2 * np.pi)
        d_v = np.linalg.norm(q)
    return b_collision, d_theta, d_v


def handle_collision_avoidance_soft(
    focal_pose, other_pose, focal_max_velocity, other_avg_velocity
):
    d = -(other_pose[:2] - focal_pose[:2]) / np.linalg.norm(
        other_pose[:2] - focal_pose[:2]
    )
    theta = np.arctan2(d[1], d[0]) % (2 * np.pi)
    eta = -other_avg_velocity * np.cos(other_pose[2] - theta) + np.sqrt(
        (other_avg_velocity**2) * (np.cos(other_pose[2] - theta) ** 2 - 1)
        + focal_max_velocity**2
    )
    q = np.array(
        [
            other_avg_velocity * np.cos(other_pose[2]) + eta * np.cos(theta),
            other_avg_velocity * np.sin(other_pose[2]) + eta * np.sin(theta),
        ]
    )
    d_theta = np.arctan2(q[1], q[0]) % (2 * np.pi)
    d_v = np.linalg.norm(q)
    return d_theta, d_v


def handle_collision_avoidance_and_caging(
    distances_squared,
    focal_pose,
    other_poses,
    nearest_pose,
    avoidance_boundary,
    max_focal_velocity,
    nearest_avg_velocity,
    target_distance_nearest,
):
    indexes = np.logical_and(
        0.0 < distances_squared, distances_squared <= avoidance_boundary**2
    )
    b_collision = np.any(indexes)
    d_theta, d_v = 0, 0
    if np.any(indexes):
        d = -np.sum(
            maths.normalize_matrix_rows(other_poses[indexes, :2] - focal_pose[:2]),
            axis=0,
        )
        theta = np.arctan2(d[1], d[0]) % (2 * np.pi)
        eta = -nearest_avg_velocity * np.cos(nearest_pose[2] - theta) + np.sqrt(
            (nearest_avg_velocity**2) * (np.cos(nearest_pose[2] - theta) ** 2 - 1)
            + max_focal_velocity**2
        )
        b_int, int_position = maths.line_circle_intersection(
            focal_pose[:2],
            focal_pose[:2]
            + max_focal_velocity
            * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32),
            nearest_pose[:2],
            target_distance_nearest,
        )
        if b_int == 1:
            # maximum distance to move in the direction of theta
            eta = min(eta, np.linalg.norm(focal_pose[:2] - int_position))
        q = np.array(
            [
                nearest_avg_velocity * np.cos(nearest_pose[2]) + eta * np.cos(theta),
                nearest_avg_velocity * np.sin(nearest_pose[2]) + eta * np.sin(theta),
            ]
        )
        d_theta = np.arctan2(q[1], q[0]) % (2 * np.pi)
        d_v = np.linalg.norm(q)
    return b_collision, d_theta, d_v


def get_nearest_distance_axis(positions,
                              distances_squared,
                              focal_point,
                              heading,
                              sign_heading,
                              absent_distance,):
    p_tilde = positions - focal_point
    heading_others = heading - np.arctan2(p_tilde[:, 1], p_tilde[:, 0])
    indexes_above = np.logical_and(
        0 < sign_heading * heading_others,
        sign_heading * heading_others <= np.pi,
    )
    indexes_under = ~indexes_above
    distance_above = distance_under = absent_distance
    if np.any(indexes_above):
        distance_above = np.sqrt(
            np.min(distances_squared[indexes_above])
        )
    if np.any(indexes_under):
        distance_under = np.sqrt(
            np.min(distances_squared[indexes_under])
        )
    return distance_above, distance_under