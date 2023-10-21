import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import animation
import json
import pandas as pd
from matplotlib.animation import FuncAnimation
from ..utils import constants, calculations
from ..setup import Config
import os
import re
import numpy as np


def prepare_plotting(figsize: tuple, y_lim: tuple):
    sn.set_theme()
    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot()
    plt.ylim(y_lim)
    plt.xticks(rotation=45)
    return axes, fig


def _save_gif(figure, anim_func, frames, path, filename):
    ani = FuncAnimation(figure, anim_func, interval=5000, frames=frames,
                        repeat=True, cache_frame_data=False)
    writer = animation.PillowWriter(fps=3, bitrate=1000)
    os.makedirs(path, exist_ok=True)
    ani.save(f"{path}/{filename}", writer=writer)


def prepare_distance_data(distance_func: calculations.Distance, dataset: pd.DataFrame, column_name: str):

    from_origin_dist = []
    from_fully_augumented_dist = []

    values = dataset[column_name].to_numpy()
    origin = values[0]
    final_conversion = values[-1]

    for index in range(len(values)):
        from_origin_dist.append(
            distance_func.count_distance(origin, values[index])
        )
        from_fully_augumented_dist.append(
            distance_func.count_distance(final_conversion, values[index])
        )
        # print(from_origin_dist)
    return from_origin_dist, from_fully_augumented_dist


def make_bar_plot(
    df: pd.DataFrame, img_id: str, save_path: str, datas_heights: tuple,
    x_labels: list, y_lim: tuple, class_name: str, file_name: str
):
    axes, fig = prepare_plotting((8, 7), y_lim)
    bar_plot = axes.bar(
        x_labels, [0]*len(x_labels), align="center"
    )

    def animate(index):
        axes.set_title("id: {class_name}_{id} noise value: {pixels}, noise %: {percentage}".format(
            class_name=class_name, id=img_id, pixels=df['noise_rate'][index], percentage=df['noise_percent'][index]))
        # print(datas_heights)
        for j in range(len(datas_heights)):
            bar_plot[j].set_height(datas_heights[j][index])

    _save_gif(fig, animate, len(datas_heights[0]) - 1,
              f"{save_path}{class_name}/{img_id}",
              f"{file_name}.gif")
    plt.close()


def make_plot(x_points: list, y_points: tuple, y_lim: tuple, title: str, save_path: str, filename: str, legend: list):
    axes, fig = prepare_plotting((8, 7), y_lim)

    for index in range(len(y_points)):
        axes.plot(x_points, y_points[index], label=legend[index])
    axes.set_title(title)
    axes.set_xlabel('noise rate')
    axes.set_ylabel('distance')
    fig.legend()
    fig.savefig(f"{save_path}/{filename}.png")


def run(config_file_path="./config.json"):
    """Runs gifs generation basing on a config
       file used to test neural network.
    """
    with open(config_file_path, "r") as file:
        config = json.load(file)
        conf = Config(config)
        for augumentation in conf.augumentations:
            path = f"./{conf.model.name.lower()}-{conf.tag}/{augumentation.name}"
            files = [os.path.join(f"{path}/dataframes", file)
                     for file in os.listdir(f"{path}/dataframes")]
            os.makedirs(f"./out/{path}", exist_ok=True)
            for file in files:
                df = pd.read_pickle(file)
                img_id = re.findall(r"\_\d+", file)[0]
                class_name = constants.LABELS_CIFAR_10.get(df['original_label'][0], "Error")
                logits = df.classifier.to_numpy()
                logits_tuple = np.array([np.squeeze(row, axis=0) for row in logits]).T

                make_bar_plot(
                    df, img_id, f"./out/{path}/",
                    logits_tuple, list(constants.LABELS_CIFAR_10.values()),
                    (-20, 20), class_name, "logits2")

                # math_scores = {k.name: [] for k in calculations.DISTANCE_FUNCS}

                for func in calculations.DISTANCE_FUNCS:

                    distances = prepare_distance_data(func, df, "features")
                    make_bar_plot(
                        df, img_id, f"./out/{path}/", distances,
                        ["origin", "augumented"], func.y_lim, class_name,
                        func.name
                    )
                    make_plot(y_points=distances, x_points=df.noise_percent.to_numpy(),
                              y_lim=func.y_lim, title=f"{class_name} {func.name} distance",
                              save_path=f"./out/{path}/{class_name}/{img_id}", filename=f"{func.name}_line_plot",
                              legend=["original", "fully transformed"]
                    )

                # with open(f"./out/{path}/{img_id}.json", "+w") as file:
                #     json.dump(math_scores, file)
