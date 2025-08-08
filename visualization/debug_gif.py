import imageio.v2 as imageio
import click

from visualization.debug import OUTPUT_FOLDER

@click.command()
@click.option("--file", type=str, default="FollowPathDebug_Gradients_seed=X")
@click.option("--time_start", type=int, default=1)
@click.option("--time_end", type=int, default=2)
@click.option("--time_step", type=int, default=1)
@click.option("--folder", type=str, default=OUTPUT_FOLDER)
def run(file="FollowPathDebug_Gradients_seed=X",
        time_start=0,
        time_end=1,
        time_step=1,
        folder=OUTPUT_FOLDER):
    out_filename = folder + file + ".gif"

    in_filenames = [folder + file + f"_t={t}.png" for t in range(time_start, time_end, time_step)]

    images = []
    for filename in in_filenames:
        images.append(imageio.imread(filename))
    # with imageio.get_writer(out_filename, mode='I') as writer:
    #     for in_filename in in_filenames:
    #         image = imageio.imread(in_filename)
    #         writer.append_data(image)
    
    imageio.mimsave(out_filename, images)

run()