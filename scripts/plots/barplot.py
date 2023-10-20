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


def show_logits(df: pd.DataFrame, save_path: str, img_id: str):
    """ Generates bar plot gif showing changes of logits

    """

    axes, fig = prepare_plotting((8, 7), (-20, 20))
    logits = df["classifier"]
    class_name = constants.LABELS_CIFAR_10.get(
        df["original_label"][0], "ERROR")
    x_labels = list(constants.LABELS_CIFAR_10.values())
    bar_plot = axes.bar(
        x_labels, [0]*len(x_labels), align="center"
    )

    def animate(index):
        value = logits[index][0]
        if isinstance(df["noise_rate"][index], float):
            rate = int(100 * df["noise_rate"][index])
        else:
            rate = df["noise_rate"][index]
        axes.set_title(
            "id: {class_name}_{id} noise_rate: {rate:04d}".format(
                class_name=class_name, id=img_id, rate=rate))

        for j in range(len(value)):
            bar_plot[j].set_height(value[j])

    _save_gif(
        fig, animate, len(logits)-1,
        f"{save_path}/{class_name}/{img_id}",
        "logits.gif"
    )
    plt.close()


def show_distances(distance_func: calculations.Distance, df: pd.DataFrame, img_id: str, save_path: str):
    """ Shows distance barplot gifs depending on used distance function.

    """

    feature_vecs = df.features.to_numpy()
    starting_image = feature_vecs[0]
    finish_image = feature_vecs[-1]
    axes, fig = prepare_plotting((8, 7), distance_func.y_lim)
    class_name = constants.LABELS_CIFAR_10.get(
        df["original_label"][0], "ERROR")
    x_labels = ["origin", "augumented"]
    bar_plot = axes.bar(
        x_labels, [0]*len(x_labels), align="center"
    )
    dist_lists = {
        "origin": [],
        "augumented": []
    }

    def animate(index):
        dist_origin = distance_func.count_distance(
            starting_image, feature_vecs[index])
        dist_finish = distance_func.count_distance(
            finish_image, feature_vecs[index])
        dist_lists["origin"].append(str(dist_origin))
        dist_lists["augumented"].append(str(dist_finish))
        if isinstance(df["noise_rate"][index], float):
            rate = int(100 * df["noise_rate"][index])
        else:
            rate = df["noise_rate"][index]
        axes.set_title("id: {class_name}_{id} noise_rate: {rate:04d}".format(
            class_name=class_name, id=img_id, rate=rate)
        )
        bar_plot[0].set_height(dist_origin)
        bar_plot[1].set_height(dist_finish)

    _save_gif(fig, animate, len(feature_vecs) - 1,
              f"{save_path}{class_name}/{img_id}",
              f"{distance_func.name}.gif")
    plt.close()
    return dist_lists


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
                show_logits(
                    df, f"./out/{path}/", img_id
                )
                math_scores = {k.name: [] for k in calculations.DISTANCE_FUNCS}
                for func in calculations.DISTANCE_FUNCS:
                    # if isinstance(func, calculations.MahalanobisDistance):
                    #     func.count_distance(
                    #         df["features"], df['features'][0]
                    #     )
                    # return
                    math_scores[func.name].append(
                        show_distances(func,  df, img_id, f"./out/{path}/")
                    )
                with open(f"./out/{path}/{img_id}.json", "+w") as file:
                    json.dump(math_scores, file)
