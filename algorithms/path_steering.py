import numba
import numpy as np

from utils import maths
from algorithms.potentials import NUMBA_NULL
from algorithms.searching import compute_search_action
from algorithms.base import (
    handle_collision_avoidance_hard,
    handle_collision_avoidance_soft,
    handle_collision_avoidance_and_caging,
)

def compute_path_steering_action(
    i,
    robot_poses,
    herd_poses,
    search_msg_all,
    steer_msg_all,
    potential,
    gradient,
    radius_robot_herd_interaction,
    circular_direction,
    rate_circular_motion,
    max_linear_velocity_robot,
    max_linear_velocity_herd,
    avg_linear_velocity_herd,
    range_detection_robot,
):
    # Experimentally chosen value
    # This velocity influences the convergence rate towards a stable caging formation
    # The higher the value, the higher the probability of oscillations (i.e. unstable formation)
    caging_velocity = 1.0

    erorr_margin_neighboring_distance = 2 * caging_velocity
    
    lower_error_margin_distance_nearest = max_linear_velocity_herd
    upper_error_margin_distance_nearest = max_linear_velocity_herd

    target_distance_nearest = (
        radius_robot_herd_interaction + lower_error_margin_distance_nearest
    )

    # bound_steer = radius_robot_herd_interaction - (
    #     max_linear_velocity_robot - max_linear_velocity_herd
    # )
    bound_steer = radius_robot_herd_interaction - max_linear_velocity_herd
    # bound_steer = radius_robot_herd_interaction - 2
    # Experimentally chosen multiplier (>1) of radius_robot_herd_interaction
    # bound_not_steer = 1.15 * radius_robot_herd_interaction + (
    #     max_linear_velocity_robot - max_linear_velocity_herd
    # )
    bound_not_steer = 1.15 * radius_robot_herd_interaction + max_linear_velocity_herd
    # bound_not_steer = 1.15 * radius_robot_herd_interaction + 2

    # bound_not_steer = 1.5 * radius_robot_herd_interaction + (
    #     max_linear_velocity_robot - max_linear_velocity_herd
    # )

    avoidance_boundary_herd = 1.0
    avoidance_boundary_robot = 1.0

    # initialize variables
    robot_pose = robot_poses[i]
    n_robots = robot_poses.shape[0]

    herd_distances_squared = np.sum((herd_poses[:, :2] - robot_pose[:2]) ** 2, axis=1)
    indexes_herd_proximity = herd_distances_squared <= range_detection_robot**2
    steer_message = np.full((n_robots, 4), NUMBA_NULL, dtype=np.float32)
    steer_message[:, 0] = 0

    output = {
        "direction_angle": 0.0,
        "direction_length": 0.0,
        "circular_direction": circular_direction,
        "search_message": 0,
        "steer_message": steer_message,
    }

    if np.any(indexes_herd_proximity):
        # Steer if there is a local herd
        output["search_message"] = 1

        # Filter herd neighbours
        herd_poses = herd_poses[indexes_herd_proximity]
        herd_distances_squared = herd_distances_squared[indexes_herd_proximity]
        index_nearest = np.argmin(herd_distances_squared)
        nearest_pose = herd_poses[index_nearest]
        distance_nearest = np.sqrt(herd_distances_squared[index_nearest])
        orientation_nearest = nearest_pose[2]

        # Filter agent neighbours
        robots_distances_squared = np.sum(
            (robot_poses[:, :2] - robot_pose[:2]) ** 2, axis=1
        )
        indexes_robots_proximity = np.logical_and(
            0.0 < robots_distances_squared,
            robots_distances_squared <= range_detection_robot**2,
        )
        robots_distances_squared = robots_distances_squared[indexes_robots_proximity]
        robot_poses = robot_poses[indexes_robots_proximity]

        # (1) STEERING - generate steering messages (before COLLISION AVOIDANCE)
        steer_message[i] = np.array(
            [steer_msg_all[i, i, 0] + 1, potential, gradient[0], gradient[1]]
        )
        potentials, gradients_x, gradients_y = [], [], []
        if not np.isinf(potential):
            potentials.append(potential)
            gradients_x.append(gradient[0])
            gradients_y.append(gradient[1])
        if np.any(indexes_robots_proximity):
            steer_messages = steer_msg_all[indexes_robots_proximity]
            for a_j in range(n_robots):
                if a_j != i:
                    a_i = np.argmax(steer_messages[:, a_j, 0])
                    steer_message[a_j] = steer_messages[a_i, a_j]
                    if not np.isinf(steer_message[a_j, 1]):
                        potentials.append(steer_message[a_j, 1])
                        gradients_x.append(steer_message[a_j, 2])
                        gradients_y.append(steer_message[a_j, 3])
        output.update({
            "steer_message": steer_message,
        })
        b_steering = len(potentials) > 0
        if b_steering:
            d_theta = np.arctan2(gradient[1], gradient[0]) % (2 * np.pi)

        # P0.0 Avoid collision with other robots
        if distance_nearest >= target_distance_nearest:
            b_collision, _theta, _v = handle_collision_avoidance_and_caging(
                distances_squared=robots_distances_squared,
                focal_pose=robot_pose,
                other_poses=robot_poses,
                nearest_pose=nearest_pose,
                avoidance_boundary=avoidance_boundary_robot,
                max_focal_velocity=max_linear_velocity_robot,
                nearest_avg_velocity=avg_linear_velocity_herd,
                target_distance_nearest=target_distance_nearest,
            )
        else:
            b_collision, _theta, _v = handle_collision_avoidance_hard(
                distances_squared=robots_distances_squared,
                focal_pose=robot_pose,
                other_poses=robot_poses,
                avoidance_boundary=avoidance_boundary_robot,
            )
        if b_collision:
            output.update(
                {
                    "direction_angle": _theta,
                    "direction_length": _v,
                }
            )
            return tuple(output.values())
        
        # P0.1 Avoid collision with the herd
        b_collision, _theta, _v = handle_collision_avoidance_hard(
            distances_squared=herd_distances_squared,
            focal_pose=robot_pose,
            other_poses=herd_poses,
            avoidance_boundary=avoidance_boundary_herd,
        )
        if b_collision:
            output.update(
                {
                    "direction_angle": _theta,
                    "direction_length": _v,
                }
            )
            return tuple(output.values())

        # (1) STEERING
        # Determine target distance based on steering status and relative positions
        heading = np.arctan2(
            robot_pose[1] - nearest_pose[1], robot_pose[0] - nearest_pose[0]
        )
        sign_heading = 1 if heading >= 0 else -1

        target_distance_nearest = bound_not_steer
        b_active_steering = False
        if b_steering:
            close_h_indexes = (
                herd_distances_squared
                <= (radius_robot_herd_interaction + max_linear_velocity_herd) ** 2
            )
            if np.any(close_h_indexes):
                gammas = np.arctan2(
                    robot_pose[1] - herd_poses[close_h_indexes, 1],
                    robot_pose[0] - herd_poses[close_h_indexes, 0],
                )
                b_active_steering = True
                for gamma in gammas:
                    b_active_steering = (
                        b_active_steering
                        and abs(
                            np.arctan2(np.sin(d_theta - gamma), np.cos(d_theta - gamma))
                        )
                        > np.pi / 2
                    )
                if b_active_steering:
                    target_distance_nearest = bound_steer
            else:
                # if robot is behind the herd and should steer into the herd
                if (
                    abs(
                        np.arctan2(np.sin(d_theta - heading), np.cos(d_theta - heading))
                    )
                    > np.pi / 2
                ):
                    target_distance_nearest = bound_steer
                    b_active_steering = True

        # P1.0 Correct errors of the robot not being at the target distance
        if (
            target_distance_nearest - distance_nearest
            >= lower_error_margin_distance_nearest
        ):
            _theta, _v = handle_collision_avoidance_soft(
                focal_pose=robot_pose,
                other_pose=nearest_pose,
                focal_max_velocity=max_linear_velocity_robot,
                other_avg_velocity=avg_linear_velocity_herd,
            )
            output.update(
                {
                    "direction_angle": _theta,
                    "direction_length": _v,
                }
            )
            return tuple(output.values())

        # (2) CAGING
        if np.any(indexes_robots_proximity):
            p_tilde = robot_poses[:, :2] - nearest_pose[:2]
            heading_others = heading - np.arctan2(p_tilde[:, 1], p_tilde[:, 0])
            indexes_above = np.logical_and(
                0 < sign_heading * heading_others,
                sign_heading * heading_others <= np.pi,
            )
            indexes_under = ~indexes_above
            distance_above = distance_under = (
                2 * radius_robot_herd_interaction - erorr_margin_neighboring_distance
            )
            if np.any(indexes_above):
                distance_above = np.sqrt(
                    np.min(
                        robots_distances_squared[
                            indexes_above
                        ]
                    )
                )
            if np.any(indexes_under):
                distance_under = np.sqrt(
                    np.min(
                        robots_distances_squared[
                            indexes_under
                        ]
                    )
                )
            # If the two nearest neighbours (one for each side defined by the heading),
            # are at the same distance from the robot (within the error margin),
            if (
                abs(distance_above - distance_under)
                <= erorr_margin_neighboring_distance / 2
            ):
                # and the robot is at the target distance within a margin of error,
                # then copy the fish's speed and direction
                if (
                    0
                    <= distance_nearest - target_distance_nearest
                    <= upper_error_margin_distance_nearest
                ):
                    output.update(
                        {
                            "direction_angle": orientation_nearest % (2 * np.pi),
                            "direction_length": avg_linear_velocity_herd,
                        }
                    )
                    return tuple(output.values())
            # Otherwise move towards to the furthest neighbour
            else:
                circular_direction = (
                    sign_heading if distance_above < distance_under else -sign_heading
                )
        # Direction of the circular motion
        caging_direction_angle = heading + circular_direction * (
            np.pi / 2
            + np.arctan(
                rate_circular_motion * (distance_nearest - target_distance_nearest)
            )
        )
        # Overwrite normal caging logic -> move to steering-radius as fast as possible!
        if b_active_steering:
            if distance_nearest - target_distance_nearest > upper_error_margin_distance_nearest:
                caging_velocity = -avg_linear_velocity_herd * np.cos(
                    orientation_nearest - caging_direction_angle
                ) + np.sqrt(
                    (avg_linear_velocity_herd**2)
                    * (np.cos(orientation_nearest - caging_direction_angle) ** 2 - 1)
                    + max_linear_velocity_robot**2
                )
        # Motion vector
        vector_herd = avg_linear_velocity_herd * maths.get_unit_vector(
            orientation_nearest
        )
        vector_caging = caging_velocity * maths.get_unit_vector(caging_direction_angle)
        q = vector_herd + vector_caging
        output.update(
            {
                "direction_angle": np.arctan2(q[1], q[0]) % (2 * np.pi),
                "direction_length": np.linalg.norm(q),
            }
        )
        return tuple(output.values())
    else:
        # Search for a local herd
        d_theta, search_message = compute_search_action(
            p=robot_poses,
            c=search_msg_all,
            i=i,
            zor=avoidance_boundary_robot,
            zoo=range_detection_robot - avoidance_boundary_robot,
            zoa=0.0,
            zoi=range_detection_robot,
        )
        output.update({
            "direction_angle": d_theta,
            "direction_length": max_linear_velocity_robot,
            "search_message": search_message,
        })
        return tuple(output.values())
