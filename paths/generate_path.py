import click
import numpy as np
import os
import time
from paths import OUTPUT_FOLDER
from paths.path import Path
import json

import argparse

from utils.geometrics import get_max_distance_reguar_polygon


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--type", type=str, default="linear")
    parser.add_argument("--total_length", type=int, default=5000)
    parser.add_argument("--segment_mean", type=int, default=200)
    parser.add_argument("--segment_std", type=int, default=50)
    parser.add_argument("--angle_mean", type=float, default=0)
    parser.add_argument("--angle_std", type=float, default=np.pi / 6)
    parser.add_argument("--angle_distribution", type=str, default="random")
    parser.add_argument("--turn_degree", type=float, default=np.pi/3)
    parser.add_argument("--margin_dx", type=float, default=500)
    parser.add_argument("--has_margin", type=bool, default=True)
    parser.add_argument("--n_mixtures", type=int, default=2)
    parser.add_argument("--gaussian_scaling", type=float, default=500000.0)
    parser.add_argument("--min_margin", type=float, default=150.0)
    parser.add_argument("--max_margin", type=float, default=500.0)
    parser.add_argument("--min_margin_multiplier", type=float, default=2.0)
    parser.add_argument("--max_margin_multiplier", type=float, default=3.0)
    parser.add_argument("--safety_margin", type=float, default=50.0)
    parser.add_argument("--safety_margin_multiplier", type=float, default=0.0)
    parser.add_argument("--cte_margin", type=float, default=None)
    parser.add_argument("--n_robots", type=int, default=5)
    parser.add_argument("--radius_robot_herd_interaction", type=float, default=20.0)
    parser.add_argument("--do_draw", type=bool, default=True)
    parser.add_argument("--do_save", type=bool, default=True)
    parser.add_argument("--sub_folder", type=str, default="")
    parser.add_argument("--out_folder", type=str, default="paths/output")

    args = parser.parse_args()
    return args


def generate_path(output_folder, variables, sub_folder=None):
    if sub_folder is None:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        sub_folder = f"{timestr}"
    total_folder = f"{output_folder}{sub_folder}"

    if not os.path.exists(total_folder):
        os.makedirs(total_folder)

    # Recompute variables
    # min_margin = get_max_distance_reguar_polygon(
    #     variables["n_robots"],
    #     variables["radius_robot_herd_interaction"]
    # )
    # variables["min_margin"] = min_margin
    variables["sub_folder"] = sub_folder
    variables["out_folder"] = output_folder


    json.dump(variables, open(f"{total_folder}/paths_parameters.json", "w"), indent=4)

    variables.pop("n_robots")
    variables.pop("radius_robot_herd_interaction")

    _generate_path(**variables)

    return sub_folder


def _generate_path(
    number,
    type,
    total_length,
    segment_mean,
    segment_std,
    angle_mean,
    angle_std,
    angle_distribution,
    turn_degree,
    margin_dx,
    has_margin,
    n_mixtures,
    gaussian_scaling,
    min_margin,
    max_margin,
    min_margin_multiplier,
    max_margin_multiplier,
    cte_margin,
    safety_margin,
    safety_margin_multiplier,
    do_draw,
    do_save,
    sub_folder,
    out_folder,
):
    for i in range(number):
        path = Path()

        if type == "linear":
            path.generate_new_linear(
                total_length=total_length,
                segment_normal_distrubtion_params=(segment_mean, segment_std),
                angle_normal_distribution_params=(angle_mean, angle_std),
                angle_distribution=angle_distribution,
            )
        elif type in ["cubic", "cubic_hermite"]:
            path.generate_new_smooth(
                smoothing=type,
                total_length=total_length,
                segment_normal_distrubtion_params=(segment_mean, segment_std),
                angle_normal_distribution_params=(angle_mean, angle_std),
                angle_distribution=angle_distribution,
            )
        elif type == "one_turn":
            path.generate_one_turn(
                turn_degree,
                margin_dx, 
                gaussian_scaling,
                min_margin,
                max_margin,
                safety_margin_multiplier,
            )
            has_margin = False
        elif type == "zero_turns":
            path.generate_zero_turns(
                total_length,
                segment_normal_distrubtion_params=(segment_mean, segment_std),
                angle_normal_distribution_params=(angle_mean, angle_std),
                angle_distribution=angle_distribution,
            )
        else:
            raise ValueError("Unknown path type: {}".format(type))

        if has_margin:
            path.generate_margin(
                n_mixtures=n_mixtures,
                gaussian_scaling=gaussian_scaling,
                min_margin=min_margin,
                max_margin=max_margin,
                min_margin_multiplier=min_margin_multiplier,
                max_margin_multiplier=max_margin_multiplier,
                cte_margin=cte_margin,
                safety_margin=safety_margin,
                safety_margin_multiplier=safety_margin_multiplier,
            )

        if do_draw:
            path.draw(out_filename=f"{sub_folder}/path_{i}", out_folder=out_folder)

        if do_save:
            path.save(out_filename=f"{sub_folder}/path_{i}", out_folder=out_folder)


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)
    generate_path(OUTPUT_FOLDER, variables)
