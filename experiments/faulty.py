import json
import os
import time
import click
import numpy as np
import argparse

from simulators.faulty import run_simulation
from visualization import debug
from models import get_model_names
from models.base import get_default_model_params
from paths.release import get_path_filename

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, default="DEBUG")
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--init_seed", type=int, default=0)
    parser.add_argument("--simulation_frequency", type=int, default=1)
    parser.add_argument("--max_time", type=int, default=2000)
    parser.add_argument("--path_init_time", type=int, default=1000)
    parser.add_argument("--save_time_step", type=int, default=1)
    parser.add_argument("--save_folder", type=str, default="")
    parser.add_argument("--n_robots", type=int, default=10)
    parser.add_argument("--n_herd", type=int, default=10)
    parser.add_argument("--path_release_id", type=str, default=None)
    parser.add_argument("--path_id", type=int, default=None)
    parser.add_argument("--path_filename", type=str, default="output/tests/test_path_margin_3")
    parser.add_argument("--model_herd_name", type=str, default="METRIC")
    parser.add_argument("--model_herd_params", "-mhp", action='append',
                        type=lambda kv: kv.split("="), default=[])
    parser.add_argument("--init_density_robot", type=float, default=0.01)
    parser.add_argument("--init_density_herd", type=float, default=0.01)
    parser.add_argument("--angular_noise_robot", type=float, default=0.05)
    parser.add_argument("--angular_noise_herd", type=float, default=0.05)
    parser.add_argument("--max_linear_velocity_robot", type=float, default=4.0)
    parser.add_argument("--max_linear_velocity_herd", type=float, default=2.0)
    parser.add_argument("--max_angular_velocity_robot", type=float, default=np.pi)
    parser.add_argument("--max_angular_velocity_herd", type=float, default=np.pi)
    parser.add_argument("--rate_circular_motion", type=float, default=0.2)
    parser.add_argument("--range_detection_robot", type=float, default=50.)
    parser.add_argument("--faults_start_time", type=int, default=0)
    parser.add_argument("--faults_interval", type=int, default=200)
    parser.add_argument("--faults_number", type=int, default=1)
    parser.add_argument("--faults_localization", type=str, default="BACK")

    args = parser.parse_args()
    return args


def run(
    run_mode="DEBUG",
    n_seeds=1,
    init_seed=0,
    simulation_frequency=1,
    max_time=2000,
    path_init_time=1000,
    save_time_step=25,
    save_folder="",
    n_robots=10,
    n_herd=10,
    path_release_id=None,
    path_id=None,
    path_filename="output/tests/test_path_margin_3",
    model_herd_name="METRIC",
    model_herd_params={},
    init_density_robot=0.01,
    init_density_herd=0.01,
    angular_noise_robot=0.05,
    angular_noise_herd=0.05,
    max_linear_velocity_robot=15.0,
    max_linear_velocity_herd=10.0,
    max_angular_velocity_robot=np.pi,
    max_angular_velocity_herd=np.pi,
    rate_circular_motion=0.2,
    range_detection_robot=50.,
    faults_start_time=0,
    faults_interval=200,
    faults_number=1,
    faults_localization="BACK",
):
    # linear_velocity is in cm/s
    # angular_velocity is in rad/s
    assert(run_mode in ["DEBUG", "PLOT", "RELEASE", "TEST_REMOTE_RUN"])
    assert(n_seeds > 0)
    assert(simulation_frequency > 0)
    assert(max_time > 0)
    assert(save_time_step > 0)
    assert(n_robots >= 0)
    assert(n_herd > 0)
    # TODO: fix assert below
    # assert(path_id in get_path_ids())
    assert(model_herd_name in get_model_names())
    assert(max_linear_velocity_robot > max_linear_velocity_herd)
    assert(faults_interval > 0)
    assert(faults_number >= 0)
    assert(faults_localization in ["BACK", "FRONT", "LEFT", "RIGHT"])

    # default model herd parameters
    model_herd_params_input = dict(model_herd_params)
    model_herd_params = get_default_model_params()
    model_herd_params.update(model_herd_params_input)
    for key in model_herd_params.keys():
        model_herd_params[key] = float(model_herd_params[key])

    # update path filename
    if (path_release_id is None) or (path_id is None):
        path_filename = f"paths/{path_filename}.pkl"
    else:
        path_filename = get_path_filename(path_release_id, path_id)

    filename = (
        "{DATA}_TYPE={TYPE}_"
        + f"NR={n_robots}_"
        + f"NH={n_herd}_"
        + f"PRI={path_release_id}_"
        + f"PI={path_id}_"
        + f"MODELH={model_herd_name}_"
        + f"ZOO={model_herd_params['zone_width_alignment']}_"
        + f"ZOA={model_herd_params['zone_width_attraction']}_"
        + f"ZOI={model_herd_params['zone_width_aversion']}_"
        + f"KOI={model_herd_params['scaling_factor_aversion']}_"
        + f"VAMAX={max_linear_velocity_robot}_"
        + f"VHMAX={max_linear_velocity_herd}_"
        + f"FST={faults_start_time}_"
        + f"FI={faults_interval}_"
        + f"FN={faults_number}_"
        + f"FL={faults_localization}_"
        + "SEED={SEED}"
    )

    if run_mode == "TEST_REMOTE_RUN":
        print(f"Running in {run_mode} mode...")
        print(f"Filename: {filename}")
        return

    debug_folder = None
    if run_mode == "DEBUG":
        timestr = time.strftime("%Y%m%d-%H%M%S")
        debug_folder = f"{debug.OUTPUT_FOLDER}{timestr}"
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)

    for seed in range(init_seed, init_seed + n_seeds):
        herd_poses, robot_poses, path, robot_activities = run_simulation(
            seed=seed,
            debug_folder=debug_folder,
            run_mode=run_mode,
            simulation_frequency=simulation_frequency,
            max_time=max_time,
            save_time_step=save_time_step,
            path_init_time=path_init_time,
            n_robots=n_robots,
            n_herd=n_herd,
            path_filename=path_filename,
            model_herd_name=model_herd_name,
            model_herd_params=model_herd_params,
            init_density_robot=init_density_robot,
            init_density_herd=init_density_herd,
            angular_noise_robot=angular_noise_robot,
            angular_noise_herd=angular_noise_herd,
            max_linear_velocity_robot=max_linear_velocity_robot,
            max_linear_velocity_herd=max_linear_velocity_herd,
            max_angular_velocity_robot=max_angular_velocity_robot,
            max_angular_velocity_herd=max_angular_velocity_herd,
            rate_circular_motion=rate_circular_motion,
            range_detection_robot=range_detection_robot,
            faults_start_time=faults_start_time,
            faults_interval=faults_interval,
            faults_number=faults_number,
            faults_localization=faults_localization,
        )

        if run_mode == "RELEASE":
            print("Saving data...")
            np.save(
                save_folder + filename.format(DATA="POSES", TYPE="H", SEED=seed) + ".npy", herd_poses
            )
            np.save(
                save_folder + filename.format(DATA="POSES", TYPE="R", SEED=seed) + ".npy", robot_poses
            )
            np.save(
                save_folder + filename.format(DATA="ACT", TYPE="J", SEED=seed) + ".npy", robot_activities
            )
            # TODO: re-do output folder names
            path.save_characteristics_json(
                out_filename=filename.format(DATA="DICT", TYPE="C", SEED=seed), out_folder=save_folder
            )
            path.save_path_pickle(
                out_filename=filename.format(DATA="CLASS", TYPE="P", SEED=seed), out_folder=save_folder
            )

        if run_mode == "PLOT":
            print("Plotting at the end...")
            # TODO: fix for new path format
            # debug.plot_poses_with_path(
            #     seed=seed,
            #     robot_poses=robot_poses,
            #     herd_poses=herd_poses,
            #     path_nodes=path_nodes,
            # )


if __name__ == '__main__':
    args = parse_args()
    run(**vars(args))

