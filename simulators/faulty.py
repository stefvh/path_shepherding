import os
import pickle
import time
import numpy as np
import models
import paths
from algorithms.path_steering import compute_path_steering_action
from algorithms.potentials import (
    NUMBA_NULL,
    get_empty_potential_gradient,
    get_follow_path_potential_gradient,
)
from simulators.kinematics import Kinematics
from visualization.debug_faults_fp import (
    plot_faults_path_follow,
    merge_plots_gif,
)
from utils import maths, labeling


def run_simulation(
    seed,
    debug_folder,
    run_mode,
    simulation_frequency,
    max_time,
    save_time_step,
    path_init_time,
    n_robots,
    n_herd,
    path_filename,
    model_herd_name,
    model_herd_params,
    init_density_robot,
    init_density_herd,
    angular_noise_robot,
    angular_noise_herd,
    max_linear_velocity_robot,
    max_linear_velocity_herd,
    max_angular_velocity_robot,
    max_angular_velocity_herd,
    rate_circular_motion,
    range_detection_robot,
    faults_start_time,
    faults_interval,
    faults_number,
    faults_localization,
):
    model_herd = models.get_model(model_herd_name, **model_herd_params)
    radius_robot_herd_interaction = model_herd_params["zone_width_aversion"]

    time_space = list(range(0, max_time, save_time_step))

    herd_poses = np.zeros((len(time_space), n_herd, 3), dtype=np.float32)
    herd_init_box_size = np.sqrt(n_herd / init_density_herd)
    herd_poses[0, :, :2] = (
        np.random.rand(n_herd, 2) * herd_init_box_size - herd_init_box_size / 2
    )
    # herd_poses[0, :, 2] = np.random.randn(
    #     n_herd,
    # ) * (2 * np.pi)
    herd_poses[0, :, 2] = 0. + np.random.normal(0, angular_noise_herd, n_herd)

    robot_poses = np.zeros((len(time_space), n_robots, 3), dtype=np.float32)
    psi_xy = np.random.randint(0, 4) * np.pi / 2  # At a random side of the herd
    psi_o = (psi_xy + np.pi) % (2 * np.pi)
    init_d = herd_init_box_size / 2 + 0.5 * range_detection_robot
    robot_poses[0, 0] = np.array(
        [np.cos(psi_xy) * init_d, np.sin(psi_xy) * init_d, psi_o], dtype=np.float32
    )
    robot_init_box_size = np.sqrt(n_robots / init_density_robot)
    robot_poses[0, 1:, 0] = np.cos(psi_xy) * (
        init_d + np.sqrt(2) * robot_init_box_size / 2
    ) + (2 * np.random.rand(n_robots - 1) - 1) * (np.sqrt(2) * robot_init_box_size / 2)
    robot_poses[0, 1:, 1] = np.sin(psi_xy) * (
        init_d + np.sqrt(2) * robot_init_box_size / 2
    ) + (2 * np.random.rand(n_robots - 1) - 1) * (np.sqrt(2) * robot_init_box_size / 2)
    robot_poses[0, 1:, 2] = psi_o + np.random.randn(n_robots - 1) * 0.1

    path = pickle.load(open(path_filename, "rb"))

    current_herd_poses = np.copy(herd_poses[0])
    current_robot_poses = np.copy(robot_poses[0])
    current_search_messages = np.full((n_robots,), 1)
    current_steer_messages = np.full(
        (n_robots, n_robots, 4), NUMBA_NULL, dtype=np.float32
    )
    current_steer_messages[:, :, 0] = 0
    current_circular_direction = np.full((n_robots,), 1)

    if run_mode == "DEBUG":
        current_gradients = np.full((n_robots, 2), NUMBA_NULL, dtype=np.float32)

    # Kinematics
    simulation_time_step = 1 / simulation_frequency
    max_linear_velocity_robot *= simulation_time_step
    max_linear_velocity_herd *= simulation_time_step
    max_angular_velocity_robot *= simulation_time_step
    max_angular_velocity_herd *= simulation_time_step
    robot_kinematics = Kinematics(
        simulation_time_step=simulation_time_step,
        angular_noise=angular_noise_robot,
        max_linear_velocity=max_linear_velocity_robot,
        max_angular_velocity=max_angular_velocity_robot,
    )
    linear_velocity_herd = 0.0
    herd_kinematics = Kinematics(
        simulation_time_step=simulation_time_step,
        angular_noise=angular_noise_herd,
        max_linear_velocity=linear_velocity_herd,
        max_angular_velocity=max_angular_velocity_herd,
    )

    avg_linear_velocity_herd = linear_velocity_herd

    # Faults
    robot_activities = np.zeros((len(time_space), n_robots), dtype=bool)
    active_robots = np.ones(n_robots, dtype=bool)
    fault_time_points = np.arange(
        faults_start_time, max_time, faults_interval
    )

    _t = 1
    print("Start time simulation...")
    for t in range(1, max_time + 1):
        if t == path_init_time:
            path_start_position = paths.computation.compute_path_start_position(
                herd_poses=current_herd_poses,
                robot_poses=current_robot_poses,
                box_size=herd_init_box_size,
                max_linear_velocity_robot=0.0,
            )
            path.reposition(path_start_position[0], path_start_position[1])
            linear_velocity_herd = max_linear_velocity_herd
            herd_kinematics = Kinematics(
                simulation_time_step=simulation_time_step,
                angular_noise=angular_noise_herd,
                max_linear_velocity=linear_velocity_herd,
                max_angular_velocity=max_angular_velocity_herd,
            )
            avg_linear_velocity_herd = linear_velocity_herd
        
        # Faults
        if t in fault_time_points:
            robot_labels = labeling.label_robot_positions(
                robot_poses=current_robot_poses,
                herd_poses=current_herd_poses,
            )
            fault_likely_robots = np.where(np.array(robot_labels) == faults_localization)[0]
            fault_likely_robots = fault_likely_robots[active_robots[fault_likely_robots]]
            if len(fault_likely_robots) > 0 and np.sum(active_robots) > 3:
                fault_robots = np.random.choice(
                    fault_likely_robots, faults_number, replace=False
                )
                for i in fault_robots:
                    active_robots[i] = False
                    current_search_messages[i] = 0
                    current_steer_messages[i] = 0
                    current_circular_direction[i] = 0

        next_robot_poses = np.zeros_like(current_robot_poses)
        next_search_messages = np.zeros_like(current_search_messages)
        next_steer_messages = np.zeros_like(current_steer_messages)
        for i in range(n_robots):
            # Faulty robots
            if not active_robots[i]:
                next_robot_poses[i] = current_robot_poses[i]
                next_search_messages[i] = 0
                next_steer_messages[i] = 0
                continue

            if t > path_init_time:
                potential, gradient = get_follow_path_potential_gradient(
                    robot_pose=current_robot_poses[i],
                    herd_poses=current_herd_poses,
                    path=path,
                )
            else:
                potential, gradient = get_empty_potential_gradient()
            (
                direction_angle,
                direction_length,
                circular_direction,
                search_message,
                steer_message,
            ) = compute_path_steering_action(
                i=i,
                robot_poses=current_robot_poses,
                herd_poses=current_herd_poses,
                search_msg_all=current_search_messages,
                steer_msg_all=current_steer_messages,
                potential=potential,
                gradient=gradient,
                radius_robot_herd_interaction=radius_robot_herd_interaction,
                circular_direction=current_circular_direction[i],
                rate_circular_motion=rate_circular_motion,
                max_linear_velocity_robot=max_linear_velocity_robot,
                max_linear_velocity_herd=max_linear_velocity_herd,
                avg_linear_velocity_herd=avg_linear_velocity_herd,
                range_detection_robot=range_detection_robot,
            )
            # herd_and_robot_poses = np.concatenate(
            #     (current_herd_poses, current_robot_poses), axis=0
            # )
            # next_robot_poses[i]  = robot_kinematics.hard_constraint(
            #     pose=current_robot_poses[i],
            #     direction_angle=direction_angle,
            #     direction_length=direction_length,
            #     other_poses=herd_and_robot_poses,
            # )
            next_robot_poses[i] = robot_kinematics.get_next_pose_by_direction(
                pose=current_robot_poses[i],
                direction_angle=direction_angle,
                direction_length=direction_length,
            )
            next_search_messages[i] = search_message
            next_steer_messages[i] = steer_message
            current_circular_direction[i] = circular_direction

            if run_mode == "DEBUG":
                current_gradients[i] = gradient

        next_herd_poses = np.zeros_like(current_herd_poses)
        avg_linear_velocity_herd = 0.0
        for i in range(n_herd):
            direction = model_herd.get_action(
                i, current_herd_poses, current_robot_poses
            )
            if direction[0] == 0 and direction[1] == 0:
                direction_angle = np.random.rand() * 2 * np.pi
                direction = linear_velocity_herd * maths.get_unit_vector(
                    direction_angle
                )
            # TODO: ...
            herd_and_robot_poses = np.concatenate(
                (current_herd_poses, current_robot_poses), axis=0
            )
            next_herd_poses[i]  = herd_kinematics.hard_constraint(
                pose=current_herd_poses[i],
                direction_angle=np.arctan2(direction[1], direction[0]),
                direction_length=np.linalg.norm(direction),
                other_poses=herd_and_robot_poses,
            )
            # next_herd_poses[i] = herd_kinematics.get_next_pose_by_direction(
            #     pose=current_herd_poses[i],
            #     direction_angle=np.arctan2(direction[1], direction[0]),
            #     direction_length=np.linalg.norm(direction),
            # )
            avg_linear_velocity_herd += np.linalg.norm(
                next_herd_poses[i] - current_herd_poses[i]
            )
        avg_linear_velocity_herd /= n_herd
        # if run_mode == "DEBUG":
        #     print(f"Average linear velocity of the herd: {avg_linear_velocity_herd:.2f}")

        current_robot_poses = np.copy(next_robot_poses)
        current_search_messages = np.copy(next_search_messages)
        current_steer_messages = np.copy(next_steer_messages)
        current_herd_poses = np.copy(next_herd_poses)

        if run_mode == "DEBUG":
            if t >= path_init_time:
                print("Debug plot at time {}".format(t))
                plot_faults_path_follow(
                    folder=f"{debug_folder}/",
                    seed=seed,
                    t=t,
                    herd_poses=current_herd_poses,
                    robot_poses=current_robot_poses,
                    robot_gradients=current_gradients,
                    path=path if t >= path_init_time else None,
                    radius_robot_herd_interaction=radius_robot_herd_interaction,
                    robot_activity=active_robots,
                )

        if t in time_space:
            robot_poses[_t] = np.copy(current_robot_poses)
            herd_poses[_t] = np.copy(current_herd_poses)
            robot_activities[_t] = np.copy(active_robots)
            _t += 1
        
        if t > path_init_time:
            if path.is_at_finish(current_herd_poses[:, :2]):
                break

    # Cut data by time period
    robot_poses = robot_poses[:_t]
    herd_poses = herd_poses[:_t]
    robot_activities = robot_activities[:_t]

    print("...Finished time simulation")

    if run_mode == "DEBUG":
        merge_plots_gif(
            folder=debug_folder,
            seed="X",
            time_start=1,
            time_end=max_time,
            time_step=5,
        )

    return herd_poses, robot_poses, path, robot_activities
