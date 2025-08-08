import numpy as np
import time
import os
import csv
import json
import sys

from paths.generate_path import generate_path
from experiments.basic import run
from analysis.command import analyse
from analysis.merge_seeds import merge

DO_PATH_CREATION = True
DO_EXPERIMENT = True
DO_PREPARE_ANALYSIS = True
DO_ANALYSIS = True
DO_PLOT = False

if len(sys.argv) > 1:
    print("Using EXPERIMENT_ID from command line")
    EXPERIMENT_ID = sys.argv[1]
else:
    print("Creating new EXPERIMENT_ID")
    EXPERIMENT_ID = time.strftime("%Y%m%d-%H%M%S")
print("EXPERIMENT_ID: ", EXPERIMENT_ID)

EXPERIMENT_OUTPUT_FOLDER = f"data/{EXPERIMENT_ID}"

PARAMETERS_OUTPUT_FOLDER = f"run-configs/{EXPERIMENT_ID}"

ANALYSIS_OUTPUT_FOLDER = f"results/{EXPERIMENT_ID}"

ANALYSIS_MERGED_OUTPUT_FOLDER = f"{ANALYSIS_OUTPUT_FOLDER}/merged"

PLOT_OUTPUT_FOLDER = f"plots/output/{EXPERIMENT_ID}"

if not os.path.exists(EXPERIMENT_OUTPUT_FOLDER):
    os.makedirs(EXPERIMENT_OUTPUT_FOLDER)

if not os.path.exists(PARAMETERS_OUTPUT_FOLDER):
    os.makedirs(PARAMETERS_OUTPUT_FOLDER)

if not os.path.exists(ANALYSIS_OUTPUT_FOLDER):
    os.makedirs(ANALYSIS_OUTPUT_FOLDER)

if not os.path.exists(ANALYSIS_MERGED_OUTPUT_FOLDER):
    os.makedirs(ANALYSIS_MERGED_OUTPUT_FOLDER)

if not os.path.exists(PLOT_OUTPUT_FOLDER):
    os.makedirs(PLOT_OUTPUT_FOLDER)


##################
# Set parameters #
##################
list_n_mixtures = [1]
list_min_margin_multiplier = [ 1.0 ]
n_path_ids = 1

default_path_params = {
    "number": n_path_ids,
    "type": "linear",
    "total_length": 5000,
    "segment_mean": 500,
    "segment_std": 100,
    "angle_mean": 0,
    "angle_std": 0,
    "angle_distribution": "random_positive",
    "has_margin": True,
    "gaussian_scaling": 500000.0,
    "min_margin": 0,
    "max_margin": 500,
    "min_margin_multiplier": 2.0,
    "max_margin_multiplier": 3.0,
    "cte_margin": None,
    "safety_margin": 0,
    "safety_margin_multiplier": 0.4,
    "do_draw": True,
    "do_save": True,
    "n_robots": 10,
    "radius_robot_herd_interaction": 20.0,
}

default_sim_params = {
    "simulation_frequency": 1,
    "max_time": 5000,
    "path_init_time": 500,
    "save_time_step": 1,
    "save_folder": f"{EXPERIMENT_OUTPUT_FOLDER}/",
    "n_robots": 6,
    "n_herd": 1,
    "path_release_id": "20230901-164440",
    "path_id": 0,
    "model_herd_name": "METRIC",
    "model_herd_params": {
        "zone_width_repulsion": 1.0,
        "zone_width_alignment": 50.0,
        "zone_width_attraction": 30.0,
        "zone_width_aversion": 40.0,
        "scaling_factor_repulsion": 100.0,
        "scaling_factor_alignment": 50.0,
        "scaling_factor_attraction": 1.0,
        "scaling_factor_aversion": 100.0,
    },
    "init_density_robot": 0.01,
    "init_density_herd": 0.01,
    "angular_noise_robot": 0.05,
    "angular_noise_herd": 0.05,
    "max_linear_velocity_robot": 4.0,
    "max_linear_velocity_herd": 2.0,
    "max_angular_velocity_robot": np.pi,
    "max_angular_velocity_herd": np.pi,
    "rate_circular_motion": 0.2,
    "range_detection_robot": 100.0,  # TODO: 50.0,
}

list_n_robots_herd = [(10, 5), (20, 30)]
min_margins = [10]
speeds = [(2, 4)]

n_seeds = 30

################
# Create paths #
################
list_path_release_id = []
if DO_PATH_CREATION:
    for n_mixtures in list_n_mixtures:
        for min_margin_multiplier in list_min_margin_multiplier:
            for min_margin in min_margins:
                # for min_margin in dict_n_robots_min_margin.values():
                params = default_path_params.copy()
                params["n_mixtures"] = n_mixtures
                params["min_margin"] = min_margin
                params["min_margin_multiplier"] = min_margin_multiplier
                path_release_id = generate_path("paths/release/", params)
                list_path_release_id.append(path_release_id)
else:
    list_path_release_id = [
        "20231007-173414",
    ]

######################
# Run the experiment #
######################
if DO_EXPERIMENT:
    time.sleep(5)
    print("Begin local EXPERIMENT run...")
    for path_release_id in list_path_release_id:
        for path_id in range(n_path_ids):
            for n_individuals in list_n_robots_herd:
                for speed in speeds:
                    params = dict(default_sim_params)
                    n_robots, n_herd = n_individuals
                    v_herd, v_robot = speed
                    params.update(
                        {
                            "path_release_id": path_release_id,
                            "path_id": path_id,
                            "n_robots": n_robots,
                            "n_herd": n_herd,
                            "max_linear_velocity_robot": v_robot,
                            "max_linear_velocity_herd": v_herd,
                        }
                    )
                    # Run
                    print(f"Running experiment with params: {params}")
                    run(
                        run_mode="RELEASE",
                        n_seeds=n_seeds,
                        init_seed=0,
                        **params,
                    )


##########################
# Analyze the experiment #
##########################
if DO_PREPARE_ANALYSIS:
    time.sleep(5)
    print("Begin local PREPARE_ANALYSIS run...")
    ## Save parameters
    with open(f"{PARAMETERS_OUTPUT_FOLDER}/params.csv", "w") as f:
        writer = csv.writer(f)

        field = ["NR", "NH", "PRI", "PI", "MODELH", "ZOO", "ZOA", "ZOI", "KOI", "VAMAX", "VHMAX"]
        writer.writerow(field)

        for path_release_id in list_path_release_id:
            for path_id in range(n_path_ids):
                for n_individuals in list_n_robots_herd:
                    n_robots, n_herd = n_individuals
                    writer.writerow(
                        [
                            n_robots,
                            n_herd,
                            path_release_id,
                            path_id,
                            default_sim_params["model_herd_name"],
                            default_sim_params["model_herd_params"][
                                "zone_width_alignment"
                            ],
                            default_sim_params["model_herd_params"][
                                "zone_width_attraction"
                            ],
                            default_sim_params["model_herd_params"][
                                "zone_width_aversion"
                            ],
                            default_sim_params["model_herd_params"][
                                "scaling_factor_aversion"
                            ],
                            default_sim_params["max_linear_velocity_robot"],
                            default_sim_params["max_linear_velocity_herd"],
                        ]
                    )

    json = json.dumps(default_sim_params)
    f = open(f"{PARAMETERS_OUTPUT_FOLDER}/params_other.json", "w")
    f.write(json)
    f.close()

    with open(f"{PARAMETERS_OUTPUT_FOLDER}/params_other.txt", "w") as f:
        f.write(
            f"ZOI_S:{default_sim_params['model_herd_params']['zone_width_aversion']}\n"
        )
        f.write(f"RHO_A:{default_sim_params['init_density_robot']}\n")
        f.write(f"RHO_S:{default_sim_params['init_density_herd']}\n")
        f.write(f"SIGMA_A:{default_sim_params['angular_noise_robot']}\n")
        f.write(f"SIGMA_S:{default_sim_params['angular_noise_herd']}\n")
        f.write(f"V_A_MAX:{default_sim_params['max_linear_velocity_robot']}\n")
        f.write(f"V_S_MAX:{default_sim_params['max_linear_velocity_herd']}\n")
        f.write(f"T_PATH_INIT:{default_sim_params['path_init_time']}\n")
        f.write(f"T_MAX:{default_sim_params['max_time']}\n")

if DO_ANALYSIS:
    time.sleep(5)
    print("Begin local ANALYSIS run...")
    ## Execute analysis
    metrics = [
        "SOM",
        # # "AM",
        # # "SM",
        # # # "SME",
        # # # "AME",
        # # # # "ST",
        # # # # "AT",
        # # # "SD",
        # # # "ATOM",
        # # # "STOM",
        "CPR",
        # # 'CP', 'CER',
        # # 'HD', 'HC', 'HO'
        # "HPO",
        # "SOMB",
        # "SOMX"
    ]
    analyse(
        params_filename=f"{PARAMETERS_OUTPUT_FOLDER}/params.csv",
        other_params_filename=f"{PARAMETERS_OUTPUT_FOLDER}/params_other.json",
        seed_start=0,
        seed_end=n_seeds,
        in_folder=f"{EXPERIMENT_OUTPUT_FOLDER}/",
        out_folder=f"{ANALYSIS_OUTPUT_FOLDER}/",
        filtered_metrics=metrics,
        local=True,
    )

    merge(
        in_folder=ANALYSIS_OUTPUT_FOLDER,
        out_folder=f"{ANALYSIS_MERGED_OUTPUT_FOLDER}/",
    )


####################
# Plot the results #
####################
if DO_PLOT:
    time.sleep(5)
    print("Begin local PLOT run...")

######################
# Output information #
#######################
print("release_path_ids: ", list_path_release_id)
print("EXPERIMENT_ID: ", EXPERIMENT_ID)
