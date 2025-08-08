import numpy as np
from utils import geometrics
from paths import computation as paths_computation

NUMBA_NULL = np.inf


def get_empty_potential_gradient():
    potential = NUMBA_NULL
    gradient = np.array([NUMBA_NULL, NUMBA_NULL])
    return potential, gradient


def get_path_potential(
    translation_length,
    translated_position,
    translated_segment,
    path_nodes,
    potential_delimiter,
):
    # The potential field is as follows
    # (1) If the fish is further away than dxy from the path, the potential is equal to the distance
    # (2) Otherwise the potential decreases in the direction of the path;
    #     The further along the path, the lower the potential.
    potential = translation_length
    if potential <= potential_delimiter:
        weight = 0
        for k in range(translated_segment + 1):
            if k < translated_segment:
                weight += -np.linalg.norm(path_nodes[k] - path_nodes[k + 1])
            else:
                weight += -np.linalg.norm(path_nodes[k] - translated_position)
        potential = potential * (1 + weight)

    return potential


def get_path_gradient(
    translation_length,
    translated_position,
    translated_segment,
    path_nodes,
    gradient_delimiter,
):
    # The gradient is as follows
    # (1) If the fish is further away than dxy from the path, the gradient is equal to the sum of
    #     (q1) the vector from the fish to the closest point on the path
    #     (q2) the vector representing the current path segment
    # (2) Otherwise the gradient is equal to (q2) the vector representing the current path segment
    if translation_length > gradient_delimiter:
        q1 = translated_position - path_nodes[translated_segment]
        q2 = path_nodes[translated_segment + 1] - path_nodes[translated_segment]
        q = 0.5 * q1 + 0.5 * q2
    else:
        q = path_nodes[translated_segment + 1] - path_nodes[translated_segment]
    gradient = q / np.linalg.norm(q)

    return gradient


def get_follow_path_potential_gradient(robot_pose, 
                                       herd_poses, 
                                       path):
    herd_dist_squared = np.sum((herd_poses[:, :2] - robot_pose[:2]) ** 2, axis=1)
    nearest_herd_pose = herd_poses[np.argmin(herd_dist_squared)]

    robot_gradient, _, is_in_safety_margin = path.get_gradient(robot_pose[0], robot_pose[1])
    herd_gradient, _, _ = path.get_gradient(nearest_herd_pose[0], nearest_herd_pose[1])

    robot_potential = path.get_potential(robot_pose[0], robot_pose[1])
    herd_potential = path.get_potential(nearest_herd_pose[0], nearest_herd_pose[1])

    potential = 0.5 * robot_potential + 0.5 * herd_potential

    if not is_in_safety_margin:
        gradient = 0.5 * robot_gradient + 0.5 * herd_gradient
    else:
        gradient = herd_gradient

    return potential, gradient


def get_path_potential_gradient(robot_pose, herd_poses, path_nodes):
    school_dist2 = np.sum((herd_poses[:, :2] - robot_pose[:2]) ** 2, axis=1)
    # h_indexes = school_dist2 <= zoi_a ** 2
    # school_dist2 = school_dist2[h_indexes]
    h_star = np.argmin(school_dist2)
    h_star_pose = herd_poses[h_star]

    dist_int, pos_int, seg_idx = paths_computation.compute_measurements_to_path(
        path_nodes, h_star_pose[:2]
    )

    potential = get_path_potential(
        translated_position=pos_int,
        translated_segment=seg_idx,
        translation_length=dist_int,
        path_nodes=path_nodes,
        potential_delimiter=0.5,
    )

    herd_box_size = geometrics.get_points_box_size(herd_poses[:, :2])
    
    gradient = get_path_gradient(
        translated_position=pos_int,
        translated_segment=seg_idx,
        translation_length=dist_int,
        path_nodes=path_nodes,
        gradient_delimiter=0.5 * (herd_box_size / 2),
    )

    return potential, gradient.astype(np.float32)
