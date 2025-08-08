import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union

from utils.plotting import Plotr
from utils.labeling import label_robot_positions

OUTPUT_FOLDER = "visualization/output/"

def _plot_poses_at_time(
    bound,
    bounded,
    n_agents,
    n_school,
    robot_poses,
    herd_poses,
    radius_robot_herd_interaction
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    color = "tab:blue"
    for g_j in range(n_school):
        g_position = herd_poses[g_j, :2]
        ax.scatter(*g_position, s=21.0, color=color, zorder=1)
        # ax.annotate(r"$f_{" + str(g_j) + "}$", g_position, fontsize=10, zorder=1)
        # TODO: comment
        # ax.quiver(
        #     g_position[0],
        #     g_position[1],
        #     np.cos(herd_poses[g_j, 2]),
        #     np.sin(herd_poses[g_j, 2]),
        #     linewidths=0.05,
        #     width=0.003,
        #     color=color,
        # )

    herd_circles = []
    for _, school_pose in enumerate(herd_poses):
        herd_circles.append(
            Point(*school_pose[:2]).buffer(radius_robot_herd_interaction)
        )
    
    herd_union = unary_union(herd_circles)
    try:
        xs, ys = herd_union.exterior.xy
        line_width_school = 0.85
        alpha = 0.8
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=line_width_school, zorder=0)
    except AttributeError:
        pass
    
    color = "tab:green"
    for r_i in range(n_agents):
        r_position = robot_poses[r_i, :2]
        ax.scatter(*r_position, s=21.0, color=color)
        # ax.annotate(r"$r_{" + str(r_i) + "}$", r_position, fontsize=10, zorder=1)
        # TODO: comment
        # ax.quiver(
        #     r_position[0],
        #     r_position[1],
        #     np.cos(robot_poses[r_i, 2]),
        #     np.sin(robot_poses[r_i, 2]),
        #     linewidths=0.05,
        #     width=0.003,
        #     color=color,
        # )

        # TODO: comment
        for r_j in range(n_agents):
            if np.linalg.norm(robot_poses[r_j, :2] - r_position) <= 2 * radius_robot_herd_interaction:
                ax.plot(
                    [r_position[0], robot_poses[r_j, 0]],
                    [r_position[1], robot_poses[r_j, 1]],
                    color=color,
                    linewidth=0.5,
                    alpha=0.5,
                )

    
    x_lim = np.array([-bound, bound])
    y_lim = np.array([-bound, bound])
    if not bounded:
        centroid = np.mean(herd_poses[:, :2], axis=0)
        x_lim += centroid[0]
        y_lim += centroid[1]
    ax.set_xlim(x_lim.tolist())
    ax.set_ylim(y_lim.tolist())

    return fig, ax
    
def _plot_poses_with_obstacles_at_time(
    bound,
    bounded,
    n_agents,
    n_school,
    robot_poses,
    herd_poses,
    obstacle_polygons,
    radius_robot_herd_interaction
):
    fig, ax = _plot_poses_at_time(
        bound=bound,
        bounded=bounded,
        n_agents=n_agents,
        n_school=n_school,
        robot_poses=robot_poses,
        herd_poses=herd_poses,
        radius_robot_herd_interaction=radius_robot_herd_interaction,
    )

    color = "tab:gray"
    for obstacle_polygon in obstacle_polygons:
        xs, ys = obstacle_polygon.exterior.xy
        ax.plot(xs, ys, color=color, alpha=0.8, linewidth=0.85, zorder=0)
    
    return fig, ax


def _plot_poses_with_path_at_time(
    bound,
    bounded,
    n_agents,
    n_school,
    robot_poses,
    herd_poses,
    path_nodes,
    radius_robot_herd_interaction,
):
    fig, ax = _plot_poses_at_time(
        bound=bound,
        bounded=bounded,
        n_agents=n_agents,
        n_school=n_school,
        robot_poses=robot_poses,
        herd_poses=herd_poses,
        radius_robot_herd_interaction=radius_robot_herd_interaction,
    )

    if path_nodes is not None:
        color = "tab:red"
        n_path_segments = path_nodes.shape[0] - 1
        for k in range(n_path_segments):
            v = path_nodes[k + 1] - path_nodes[k]
            ax.arrow(
                path_nodes[k, 0],
                path_nodes[k, 1],
                v[0],
                v[1],
                color=color,
                head_width=0.1,
                head_length=0.1,
            )
    
    return fig, ax


def plot_poses_with_obstacles(
    seed,
    bound,
    bounded,
    herd_poses,
    robot_poses,
    obstacle_polygons,
    radius_robot_herd_interaction=20.0,
):
    t_max = herd_poses.shape[0]
    n_agents = robot_poses.shape[1]
    n_school = herd_poses.shape[1]

    for t in range(1, t_max):
        fig, _  = _plot_poses_with_obstacles_at_time(
            bound=bound,
            bounded=bounded,
            n_agents=n_agents,
            n_school=n_school,
            robot_poses=robot_poses[t],
            herd_poses=herd_poses[t],
            obstacle_polygons=obstacle_polygons,
            radius_robot_herd_interaction=radius_robot_herd_interaction,
        )
        filename = f"ObstacleDebug_seed={seed}_t={t}.png"
        fig.savefig(OUTPUT_FOLDER + filename)
        plt.close()

def plot_poses_with_obstacles_at_time(
    seed,
    t,
    bound,
    bounded,
    herd_poses,
    robot_poses,
    obstacle_polygons,
    radius_robot_herd_interaction=20.0,
):
    n_agents = robot_poses.shape[0]
    n_school = herd_poses.shape[0]

    fig, ax = _plot_poses_with_obstacles_at_time(
        bound=bound,
        bounded=bounded,
        n_agents=n_agents,
        n_school=n_school,
        robot_poses=robot_poses,
        herd_poses=herd_poses,
        obstacle_polygons=obstacle_polygons,
        radius_robot_herd_interaction=radius_robot_herd_interaction,
    )

    filename = f"ObstacleDebug_seed={seed}_t={t}.png"
    fig.savefig(OUTPUT_FOLDER + filename)
    plt.close()

def plot_poses_with_path(
    seed,
    herd_poses,
    robot_poses,
    path_nodes,
    radius_robot_herd_interaction=20.0,
):
    bound = 60.0

    t_max = herd_poses.shape[0]
    n_agents = robot_poses.shape[1]
    n_school = herd_poses.shape[1]

    for t in range(1, t_max):
        fig, _  = _plot_poses_with_path_at_time(
            bound=bound,
            bounded=False,
            n_agents=n_agents,
            n_school=n_school,
            robot_poses=robot_poses[t],
            herd_poses=herd_poses[t],
            path_nodes=path_nodes,
            radius_robot_herd_interaction=radius_robot_herd_interaction,
        )
        filename = f"FollowPathDebug_seed={seed}_t={t}.png"
        fig.savefig(OUTPUT_FOLDER + filename)
        plt.close()

def _plot_gradients_at_time(
    ax,
    n_agents,
    robot_poses,
    robot_gradients,
):
    for r_i in range(n_agents):
        r_position = robot_poses[r_i, :2]
        gradient = robot_gradients[r_i, :2]
        if not np.isinf(gradient).any():
            ax.quiver(r_position[0], r_position[1], gradient[0], gradient[1],
                                  linewidths=0.5, width=0.003, color="tab:purple")
    return ax

def plot_poses_with_path_and_gradients_at_time(
    seed,
    t,
    herd_poses,
    robot_poses,
    robot_gradients,
    path_nodes,
    radius_robot_herd_interaction=20.0,
):
    bound = 200.0 # 60.0

    n_agents = robot_poses.shape[0]
    n_school = herd_poses.shape[0]

    fig, ax = _plot_poses_with_path_at_time(
        bound=bound,
        bounded=False,
        n_agents=n_agents,
        n_school=n_school,
        robot_poses=robot_poses,
        herd_poses=herd_poses,
        path_nodes=path_nodes,
        radius_robot_herd_interaction=radius_robot_herd_interaction,
    )

    ax = _plot_gradients_at_time(
        ax=ax,
        n_agents=n_agents,
        robot_poses=robot_poses,
        robot_gradients=robot_gradients,
    )
            
    filename = f"FollowPathDebug_Gradients_seed={seed}_t={t}.png"
    fig.savefig(OUTPUT_FOLDER + filename)
    plt.close()


def _plot_obstacles_at_time(
    ax,
    obstacle_dict,
):
    obstacle_color = "tab:red"
    obstacle_polygons_exterior = obstacle_dict["polygons_exterior"]
    for obstacle_polygon_exterior in obstacle_polygons_exterior:
        x, y = obstacle_polygon_exterior.xy
        ax.plot(x, y, linewidth=2, color=obstacle_color)
    return ax


def plot_poses_with_scenario_and_gradients_at_time(
    seed,
    t,
    herd_poses,
    robot_poses,
    robot_gradients,
    scenario,
    radius_robot_herd_interaction=20.0,
):
    if scenario["bounded"]:
        bound = scenario["bound"]
    else:
        bound = 200.0 # 60.0

    n_agents = robot_poses.shape[0]
    n_school = herd_poses.shape[0]

    fig, ax = _plot_poses_with_path_at_time(
        bound=bound,
        bounded=scenario["bounded"],
        n_agents=n_agents,
        n_school=n_school,
        robot_poses=robot_poses,
        herd_poses=herd_poses,
        path_nodes=scenario["path"],
        radius_robot_herd_interaction=radius_robot_herd_interaction,
    )

    if robot_gradients is not None:
        ax = _plot_gradients_at_time(
            ax=ax,
            n_agents=n_agents,
            robot_poses=robot_poses,
            robot_gradients=robot_gradients,
        )

    ax = _plot_obstacles_at_time(
        ax=ax,
        obstacle_dict=scenario["obstacle_dict"],
    )

    filename = f"ScenarioDebug_Gradients_seed={seed}_t={t}.png"
    fig.savefig(OUTPUT_FOLDER + filename)
    plt.close()

def plot_poses_with_scenario(
    seed,
    herd_poses,
    robot_poses,
    scenario,
    radius_robot_herd_interaction=20.0,
):
    t_max = herd_poses.shape[0]

    for t in range(1, t_max):
        fig, _  = plot_poses_with_scenario_and_gradients_at_time(
            seed=seed,
            t=t,
            herd_poses=herd_poses[t],
            robot_poses=robot_poses[t],
            robot_gradients=None,
            scenario=scenario,
            radius_robot_herd_interaction=radius_robot_herd_interaction,
        )
        filename = f"FollowPathDebug_seed={seed}_t={t}.png"
        fig.savefig(OUTPUT_FOLDER + filename)
        plt.close()