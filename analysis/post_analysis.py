import json
import pickle

import numpy as np
import pandas as pd


def get_filename(params):
    filename_template = "NR=" + str(params["NR"]) + "_"
    filename_template += "NH=" + str(params["NH"]) + "_"
    filename_template += "PRI=" + params["PRI"] + "_"
    filename_template += "PI=" + str(params["PI"]) + "_"
    filename_template += "MODELH=" + params["MODELH"] + "_"
    filename_template += "ZOO=" + str(params["ZOO"]) + "_"
    filename_template += "ZOA=" + str(params["ZOA"]) + "_"
    filename_template += "ZOI=" + str(params["ZOI"]) + "_"
    filename_template += "KOI=" + str(params["KOI"]) + "_"
    filename_template += "VAMAX=" + str(params["VAMAX"]) + "_"
    filename_template += "VHMAX=" + str(params["VHMAX"]) + "_"

    return filename_template

def get_filename_incl_faults(params):
    filename_template = get_filename(params)
    filename_template += "FST=" + str(params["FST"]) + "_"
    filename_template += "FI=" + str(params["FI"]) + "_"
    filename_template += "FN=" + str(params["FN"]) + "_"
    filename_template += "FL=" + params["FL"] + "_"

    return filename_template

def get_data_per_seed(experiment_id, seed, params, construct_filename):
    filename_template = construct_filename(params)
    filename_template += "SEED=" + str(seed) 

    robot_poses = np.load(
        open(
            f"data/{experiment_id}/POSES_TYPE=R_{filename_template}.npy",
            "rb",
        )
    )
    herd_poses = np.load(
        open(
            f"data/{experiment_id}/POSES_TYPE=H_{filename_template}.npy",
            "rb",
        )
    )
    path = pickle.load(
        open(
            f"data/{experiment_id}/CLASS_TYPE=P_{filename_template}.pkl",
            "rb",
        )
    )
    return robot_poses, herd_poses, path


def filter_df_by_fault_vars(df, params):
    df = df.loc[
        (df["FST"] == params["FST"])
        & (df["FI"] == params["FI"])
        & (df["FN"] == params["FN"])
        & (df["FL"] == params["FL"])
    ]
    return df


def filter_df_by_speed_vars(df, params):
    df = df.loc[
        (df["VAMAX"] == params["VAMAX"])
        & (df["VHMAX"] == params["VHMAX"])
    ]
    return df


def xbased_cpr(experiment_id, original_path, n_seeds, params, filter_df, construct_filename):
    data = {
        "seeds": [],
        "x": [],
        "value": [],
    }
    t_start = 0

    batch = 10
    x_min = round(original_path.x_range[0], -1)
    x_max = round(original_path.x_range[1], -1)
    length = int((x_max - x_min) / batch) + 1

    nh = params["NH"]

    cpr = pickle.load(open(f"results/{experiment_id}/merged/CPR_perseed.pkl", "rb"))

    cpr_1 = filter_df(cpr, params)

    for seed in range(n_seeds):
        cpr_2 = cpr_1.loc[cpr_1["seed"] == seed]

        robot_poses, herd_poses, path = get_data_per_seed(experiment_id, seed, params, construct_filename)

        values = {}

        x_buffer = path.x_range[0] - original_path.x_range[0]

        for point in range(length):
            values[point] = []
        
        cpr_3 = cpr_2["Measurement"].values

        t_max = robot_poses.shape[0]
        for t in range(t_start, t_max, 1):
            if np.any(path.x_range[1] - 200.0 < herd_poses[t, :, 0]):
                break
            else:
                value = 0.0
                # Is caged ?
                cpr_value = cpr_3[t]
                value = 1 - cpr_value

            point = int((np.mean(herd_poses[t, :, 0]) - x_buffer - x_min) / batch)
            if 0 <= point <= length - 1:
                values[point].append(value)

        for point in range(length):
            if len(values[point]) > 0:
                value = np.min(values[point])
                data.update(
                    {
                        "seeds": data["seeds"] + [seed],
                        "x": data["x"] + [point * batch + x_min],
                        "value": data["value"] + [value],
                    }
                )

    return pd.DataFrame(data)


def xbased_som(experiment_id, original_path, n_seeds, params, filter_df, construct_filename):
    data = {
        "seeds": [],
        "x": [],
        "value": [],
    }
    t_start = 0

    batch = 10
    x_min = round(original_path.x_range[0], -1)
    x_max = round(original_path.x_range[1], -1)
    length = int((x_max - x_min) / batch) + 1

    nh = params["NH"]

    for seed in range(n_seeds):
        robot_poses, herd_poses, path = get_data_per_seed(experiment_id, seed, params, construct_filename)

        values = {}

        x_buffer = path.x_range[0] - original_path.x_range[0]

        for point in range(length):
            values[point] = []

        t_max = robot_poses.shape[0]
        for t in range(t_start, t_max, 1):
            if np.any(path.x_range[1] - 200.0 < herd_poses[t, :, 0]):
                break
            else:
                value = 0.0
                for i in range(nh):
                    # Is in margin ?
                    is_in_margin = path.is_in_margin(*herd_poses[t, i, :2])

                    value += is_in_margin
                value /= nh
            point = int((np.mean(herd_poses[t, :, 0]) - x_buffer - x_min) / batch)
            if 0 <= point <= length - 1:
                values[point].append(value)

        for point in range(length):
            if len(values[point]) > 0:
                value = np.min(values[point])
                data.update(
                    {
                        "seeds": data["seeds"] + [seed],
                        "x": data["x"] + [point * batch + x_min],
                        "value": data["value"] + [value],
                    }
                )

    return pd.DataFrame(data)


def get_original_path(pri, pi):
    original_path = pickle.load(open(f"paths/release/{pri}/path_{pi}.pkl", "rb"))
    return original_path


def get_nseeds(experiment_id):
    dataframe = pd.read_pickle(f"results/{experiment_id}/merged/CPR_perseed.pkl")
    return len(dataframe["seed"].unique())


def group_df(df, group_name):
    df_mean = df.groupby(group_name).mean().reset_index()
    df_std = df.groupby(group_name).std().reset_index()

    df_grouped = df_mean.join(df_std["value"], rsuffix="_std")
    df_grouped.rename(columns={"value": "Mean", "value_std": "Std"}, inplace=True)

    return df_grouped


def post_analysis_xbased(experiment_id, xbased_func, filter_func, filename_func, dict_params_fixed, list_observed_params, output_name, plot_output_folder):
    for observed_params in list_observed_params:
        params = dict_params_fixed.copy()
        params.update(observed_params)

        n_seeds = get_nseeds(experiment_id)
        original_path = get_original_path(params["PRI"], params["PI"])

        df_perseed = xbased_func(experiment_id, original_path, n_seeds, params, filter_func, filename_func)
        df_full = group_df(df_perseed, "x")

        filename_sub = "_".join([f"{key}={value}" for key, value in observed_params.items()])
        df_full.to_pickle(f"{plot_output_folder}/{output_name}_{filename_sub}.pkl")


def get_mean_n_active_robots(experiment_id, params,):
    mean = 0

    n_seeds = get_nseeds(experiment_id)
    for seed in range(n_seeds):
        filename_template = get_filename(seed, params)
        props = json.load(open(f"data/{experiment_id}/PROPS_TYPE=J_{filename_template}.json", "r"))
        n_active_robots = props["n_active_robots"]
        mean += n_active_robots
    
    mean /= n_seeds
    return mean

def get_robot_activity(experiment_id, params):
    n_seeds = get_nseeds(experiment_id)
    robot_activities = []
    for seed in range(n_seeds):
        filename_template = get_filename(seed, params)
        robot_activity = np.load(
        open(
            f"data/{experiment_id}/ACT_TYPE=J_{filename_template}.npy", "rb",
            )
        )
        robot_activities.append(robot_activity[:3000])
    robot_activities = np.array(robot_activities)
    robot_activities = np.sum(robot_activities, axis=2)
    robot_activities_mean = np.mean(robot_activities, axis=0)
    robot_activities_std = np.std(robot_activities, axis=0)
    return robot_activities_mean, robot_activities_std
