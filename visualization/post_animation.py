
from visualization.debug import (
    _plot_poses_at_time,
    _plot_gradients_at_time,
    OUTPUT_FOLDER,
)
from matplotlib import pyplot as plt
from paths import _draw_safety_margin

TEMPLATE_FILENAME = "FollowPathDebug_Gradients_seed={seed}_t={t}"

def _plot_smooth_path_at_time(
    ax,
    path,
):
    nodes = path.nodes
    x_values = path.x_space
    y_values = path.y_values

    # _, x_local_optimum, y_local_optimum, _ = path.get_characteristics()
    # ax.plot(x_local_optimum, y_local_optimum, "o", label="optimum", color="tab:orange")

    # ax.plot(nodes[:, 0], nodes[:, 1], "o", label="nodes", color="tab:red")
    # ax.plot(x_values, y_values, label="spline", color="tab:blue")

    color_margin = "tab:purple"
    ax.plot(
        path.top_margin_x_space,
        path.top_margin_y_space,
        color=color_margin,
        label="top margin",
    )

    ax.plot(
        path.bot_margin_x_space,
        path.bot_margin_y_space,
        color=color_margin,
        label="bot margin",
    )

    return ax


def plot_path_follow(
    folder,
    seed,
    t_start,
    t_end,
    t_step,
    herd_poses,
    robot_poses,
    path,
    radius_robot_herd_interaction,
):
    for t in range(t_start, t_end, t_step):
        n_agents = robot_poses.shape[1]
        n_school = herd_poses.shape[1]

        fig, ax = _plot_poses_at_time(
            bound=500.0,
            bounded=False,
            n_agents=n_agents,
            n_school=n_school,
            robot_poses=robot_poses[t],
            herd_poses=herd_poses[t],
            radius_robot_herd_interaction=radius_robot_herd_interaction,
        )

        if path is not None:
            ax = _plot_smooth_path_at_time(
                ax=ax,
                path=path,
            )

            # ax = _draw_safety_margin(
            #     ax=ax,
            #     path=path,
            # )

        fig.savefig(folder + TEMPLATE_FILENAME.format(seed=seed, t=t) + ".png")
        plt.close()