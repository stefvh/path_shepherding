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

# TODO: finish this algorithm


def compute_density_maintain_action(
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
):
    lower_error_margin_distance_nearest = max_linear_velocity_herd
    upper_error_margin_distance_nearest = max_linear_velocity_herd

    target_distance_nearest = (
        radius_robot_herd_interaction + lower_error_margin_distance_nearest
    )

    # Experimentally chosen value
    # This parameter influences the convergence rate towards a stable caging formation
    # The higher the value, the higher the probability of oscillations (i.e. unstable formation)
    caging_velocity = 1.0

    erorr_margin_neighboring_distance = 2 * caging_velocity

    detection_range = 50.

    zor_h = 1.0
    zor_a = 1.0
    zoo_a = detection_range - zor_a
    zoa_a = 0.0

    # initialize variables
    robot_pose = robot_poses[i]
    n_robots = robot_poses.shape[0]

    herd_distances_squared = np.sum((herd_poses[:, :2] - robot_pose[:2]) ** 2, axis=1)
    indexes_herd_proximity = herd_distances_squared <= detection_range**2
    # maintain_message = np.full((n_robots, 4), NUMBA_NULL, dtype=np.float32)
    # maintain_message[:, 0] = 0

    output = {
        "direction_angle": 0.0,
        "direction_length": 0.0,
        "circular_direction": circular_direction,
        "search_message": 0,
    }

    if np.any(indexes_herd_proximity):
        output["search_message"] = 1

        # Filter herd neighbours
        herd_poses = herd_poses[indexes_herd_proximity]
        herd_distances_squared = herd_distances_squared[indexes_herd_proximity]
        index_nearest = np.argmin(herd_distances_squared)
        nearest_pose = herd_poses[index_nearest]
        distance_nearest = np.sqrt(herd_distances_squared[index_nearest])
        orientation_nearest = nearest_pose[2]

        # Filter robot neighbours
        robots_distances_squared = np.sum(
            (robot_poses[:, :2] - robot_pose[:2]) ** 2, axis=1
        )
        indexes_robots_proximity = np.logical_and(
            0.0 < robots_distances_squared,
            robots_distances_squared <= detection_range**2,
        )
        robot_poses = robot_poses[indexes_robots_proximity]

        # P0.0 Avoid collision with other agents
        if distance_nearest >= target_distance_nearest:
            b_collision, _theta, _v = handle_collision_avoidance_and_caging(
                distances_squared=robots_distances_squared,
                focal_pose=robot_pose,
                other_poses=robot_poses,
                nearest_pose=nearest_pose,
                avoidance_boundary=zor_a,
                max_focal_velocity=max_linear_velocity_robot,
                nearest_avg_velocity=avg_linear_velocity_herd,
                target_distance_nearest=target_distance_nearest,
            )
        else:
            b_collision, _theta, _v = handle_collision_avoidance_hard(
                distances_squared=robots_distances_squared,
                focal_pose=robot_pose,
                other_poses=robot_poses,
                avoidance_boundary=zor_a,
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
            avoidance_boundary=zor_h,
        )
        if b_collision:
            output.update(
                {
                    "direction_angle": _theta,
                    "direction_length": _v,
                }
            )
            return tuple(output.values())

        # P?? Correct errors of the robot not being at the target distance
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

        # P?? Caging
        if np.any(indexes_robots_proximity):
            heading = np.arctan2(
                robot_pose[1] - nearest_pose[1], robot_pose[0] - nearest_pose[0]
            )
            sign_heading = 1 if heading >= 0 else -1
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
                        robots_distances_squared[indexes_robots_proximity][
                            indexes_above
                        ]
                    )
                )
            if np.any(indexes_under):
                distance_under = np.sqrt(
                    np.min(
                        robots_distances_squared[indexes_robots_proximity][
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
        d_theta, search_message = compute_search_action(
            p=robot_poses,
            c=search_msg_all,
            i=i,
            zor=2 * erorr_margin_neighboring_distance,
            zoo=zoo_a,
            zoa=zoa_a,
            zoi=detection_range,
        )
        d_v = max_linear_velocity_robot
        output.update(
            {
                "direction_angle": d_theta,
                "direction_length": d_v,
                "search_message": search_message,
            }
        )
        return tuple(output.values())
