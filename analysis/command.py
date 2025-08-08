import argparse
import glob
import pickle
import re
import json

import click
import numpy as np
import pandas as pd

from analysis import metrics_caging, metrics_couzin, metrics_path
from utils import sorting


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_filename", type=str, default="params.csv")
    parser.add_argument("--other_params_filename", type=str, default="params_other.csv")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=2)
    parser.add_argument("--in_folder", type=str, default="")
    parser.add_argument("--out_folder", type=str, default="")

    args = parser.parse_args()
    return args


def analyse(params_filename="",
            other_params_filename="",
            seed_start=0,
            seed_end=1,
            in_folder="",
            out_folder="",
            filtered_metrics=[],
            local=False):
    # retrieve parameter values
    with open(other_params_filename, 'r') as f:
        other_params = json.load(f)

    if local:
        zone_width_aversion = other_params["model_herd_params"]["zone_width_aversion"]
    else:
        zone_width_aversion = float(other_params["ZOI_S"])

    _metrics = []

    # path metrics
    _metrics.extend([
        _metric('SM', 'follow_path_margin', metrics_path.metric_school_in_margin(), True, 0),
        _metric('AM', 'follow_path_margin', metrics_path.metric_agents_in_margin(), True, 0),
        _metric('ST', 'follow_path_margin', metrics_path.metric_time_school_finished_path(), False, 0),
        _metric('AT', 'follow_path_margin', metrics_path.metric_time_agents_finished_path(), False, 0),
        _metric('SD', 'follow_path_margin', metrics_path.metric_school_in_direction_path(), True, 0),
        _metric('SDB', 'follow_path_margin', metrics_path.metric_school_in_direction_batched_path(10), True, 0, 10),
        _metric('SME', 'follow_path_margin', metrics_path.metric_school_margin_error(), True, 0),
        _metric('AME', 'follow_path_margin', metrics_path.metric_agents_margin_error(), True, 0),
        _metric('ATOM', 'follow_path_margin', metrics_path.metric_agents_time_out_of_margin(), False, 0),
        _metric('STOM', 'follow_path_margin', metrics_path.metric_school_time_out_of_margin(), False, 0),
        _metric('SOM', 'follow_path_margin', metrics_path.metric_school_out_margin(), True, 0, 1),
        _metric('HPO', 'follow_path_margin', metrics_path.metric_herd_path_orientation(2 * zone_width_aversion), True, 0, 25),
        _metric('SOMB', 'follow_path_margin', metrics_path.metric_school_out_margin_batched(25), True, 0, 25),
        _metric('SOMX', 'follow_path_margin', metrics_path.metric_school_out_margin_x_based(), False, 0, 1, True),
    ])

    # additional caging metrics
    _metrics.extend([
        _metric('CP', 'caging', metrics_caging.metric_caging_probability(2 * zone_width_aversion), True, 0),
        _metric('CER', 'caging', metrics_caging.metric_convex_enclosure_rate(), True, 0),
        _metric('CPR', 'caging', metrics_caging.metric_caging_rate(2 * zone_width_aversion), True, 0, 1),
    ])

    # additional couzin metrics
    _metrics.extend([
        _metric('HD', 'couzin', metrics_couzin.metric_herd_density(), True, 0),
        _metric('HC', 'couzin', metrics_couzin.metric_herd_connectivity(2 * zone_width_aversion), True, 0),
        _metric('HO', 'couzin', metrics_couzin.metric_herd_order(), True, 0)
    ])

    params_df = pd.read_csv(params_filename)
    variable_names = list(params_df.columns)

    for metric in _metrics:
        if metric['of'] in filtered_metrics:
            print("Analyzing metric: ", metric['of'])
            measure_data(in_folder=in_folder,
                        out_folder=out_folder,
                        metric=metric,
                        variable_names=variable_names,
                        seed_start=seed_start,
                        seed_end=seed_end)


def _metric(of, model, func, time_variant, t_0, t_step=1, x_variant=False):
    return {'of': of, 'model': model, 'func': func, 'time_variant': time_variant, 't_0': t_0, 't_step': t_step, 'x_variant': x_variant}


def measure_data(in_folder,
                 out_folder,
                 metric,
                 variable_names,
                 seed_start,
                 seed_end):
    files = {"R": [], "H": [], "P": []}
    for file_type in files.keys():
        filenames = glob.glob(f'{in_folder}*TYPE={file_type}*')
        filenames.sort(key=sorting.natural_keys)
        files[file_type] = filenames
    data = {'Measurement': [], 'seed': []}
    if metric['x_variant']:
        data['x'] = []
    if metric['time_variant']:
        data['time'] = []
    for var_name in variable_names:
        data[var_name] = []

    if len(files["R"]) < 1:
        print("NO FILES")
        return
    if len(files["R"]) != len(files["H"]) != len(files["P"]):
        print("MISSING FILES")
        return
    n_files = len(files["R"])

    zone_models = ['follow_path', 'follow_path_margin']

    for i in range(n_files):
        seed = int(re.search('SEED=([0-9]*).npy', files["R"][i]).group(1))
        if seed in range(seed_start, seed_end):
            try:
                data_array_agent = np.load(files["R"][i])
                data_array_school = np.load(files["H"][i])
                if metric['model'] == 'follow_path':
                    data_array_zone = np.load(files["P"][i])
                if metric['model'] == 'follow_path_margin':
                    data_array_zone = pickle.load(open(files["P"][i], "rb"))
            except IndexError:
                print("IndexError")
            except ValueError:
                print("ValueError")

            metric_func = metric['func']

            if metric['x_variant']:
                x_values, y_values = metric_func(data_array_agent, data_array_school, data_array_zone)
                data['x'].extend(x_values)
                data['Measurement'].extend(y_values)
                data['seed'].extend([seed] * len(x_values))
                for var_name in variable_names:
                    if re.search(f'{var_name}=', files["R"][i]):
                        val_postfix = files["R"][i].split(f'{var_name}=')[1]
                        var_value = val_postfix.split('_')[0]
                        if var_value.isnumeric():
                            try:
                                value = int(var_value)
                            except ValueError:
                                value = float(var_value)
                        else:
                            value = var_value
                        data[var_name].extend([value] * len(x_values))
            else:
                t_max = data_array_agent.shape[0]
                if metric['time_variant']:
                    t_0 = metric['t_0']
                    time_points = list(range(t_0, t_max, metric['t_step']))
                    for t in time_points:
                        if metric['model'] in zone_models:
                            data_point = metric_func(t, data_array_agent, data_array_school, data_array_zone)
                        else:
                            data_point = metric_func(t, data_array_agent, data_array_school)
                        data['Measurement'].append(data_point)
                        data['time'].append(t)
                        data['seed'].append(seed)
                        for var_name in variable_names:
                            if re.search(f'{var_name}=', files["R"][i]):
                                val_postfix = files["R"][i].split(f'{var_name}=')[1]
                                var_value = val_postfix.split('_')[0]
                                if var_value.isnumeric():
                                    try:
                                        value = int(var_value)
                                    except ValueError:
                                        value = float(var_value)
                                else:
                                    value = var_value
                                data[var_name].append(value)
                else:
                    if metric['model'] in zone_models:
                        data_point = metric_func(data_array_agent, data_array_school, data_array_zone)
                    else:
                        data_point = metric_func(data_array_agent, data_array_school)
                    data['Measurement'].append(data_point)
                    data['seed'].append(seed)
                    for var_name in variable_names:
                        if re.search(f'{var_name}=', files["R"][i]):
                            val_postfix = files["R"][i].split(f'{var_name}=')[1]
                            var_value = val_postfix.split('_')[0]
                            if var_value.isnumeric():
                                try:
                                    value = int(var_value)
                                except ValueError:
                                    value = float(var_value)
                            else:
                                value = var_value
                            data[var_name].append(value)
                data_array_agent, data_array_school = None, None
                if metric['model'] in zone_models:
                    data_array_zone = None

    # Store data
    for var_name in variable_names:
        if len(data[var_name]) == 0:
            data.pop(var_name)
    df = pd.DataFrame(data=data)

    out_fix = files["R"][0].split('TYPE=R_')
    # prefixes = out_fix[0]
    # prefix = prefixes.split("/")[-1]
    prefix = ""
    suffix = out_fix[1].split('SEED')[0]
    for var_name in variable_names:
        suffix = re.sub(f'^{var_name}=[^_]*_', '', suffix)
        suffix = re.sub(f'_{var_name}=[^_]*_', '_', suffix)
    output_filepath = out_folder + prefix + suffix + metric["of"] + f'_{seed_start}-{seed_end}.pkl'
    df.to_pickle(output_filepath)


if __name__ == '__main__':
    args = parse_args()
    analyse(**vars(args))
