import argparse
from os import makedirs, path 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import re

LABELS_CIFAR_10 = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def prepare_script() -> (argparse.Namespace, str):
    parser = argparse.ArgumentParser(
        description="Script to visualize distribution of classifiers."
    )
    parser.add_argument(
        "-f", "--file", required=True, help="input file. Should be pickle."
    )
    parser.add_argument("-o", "--output", required=True, help="Dir to save generated file")
    args = parser.parse_args()
    output_title = re.findall(r"\w+\.", args.file)

    return args, output_title[0]


def make_visualization(df: pd.DataFrame, labels: dict, plot_title: str,
                       save_dir: str | None = None) -> None:
    """Visualize PCA

    Draw model's class distribution on a scatter plot.

    Parameters
    ----------
    df: DataFrame
        Dataframe containing PCA and original labels keys
    labels: dict
        Labels for each of possible classifier values.
    save_dir: str, None = False
        Save directory for generated image, if none was provided, just display it
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("PCA 1", fontsize=15)
    ax.set_ylabel("PCA 2", fontsize=15)
    ax.set_title(plot_title, fontsize=20)

    for label in labels.keys():
        indices_to_keep = df["original_label"] == label
        ax.scatter(df.loc[indices_to_keep, "PC1"], df.loc[indices_to_keep, "PC2"], s=50)
    ax.legend(labels.values())
    ax.grid()
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.plot()


def prepare_pca(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset for PCA visualization.

    Make dimension reduction to 2D by using PCA algorithm.

    Parameters
    -----
    dataframe: DataFrame
        Dataframe, that holds features and labels of deep learning model.
    Returns
    -------
    Dataframe containing original labels and reduced dimensions of features.

    """
    pca = PCA(n_components=2)
    features = np.vstack(dataframe.features.values)
    X_reduced = pca.fit_transform(features)
    df = pd.DataFrame(X_reduced, columns=["PC1", "PC2"])
    final_df = pd.concat([df, dataframe.original_label], axis=1)
    print(final_df.original_label.value_counts())
    return final_df


def run(input_dataframe_path: str, plot_title: str,
        labels: dict | None = LABELS_CIFAR_10,
        output_path: str | None = None) -> None:
    """Util function to run PCA generation
    
    Parameters
    ----------
    input_dataframe_path: str
        Path to pickle file
    plot_title: str
        Name of generated plot
    labels: dict, optional
        Labels describing keys in the dataframe. If not provided it will use CIFAR_10 labels
    output_file: str, optional
        Directory to the file where it should save plot.
    """    
    dataframe = pd.read_pickle(input_dataframe_path)
    PCA_df = prepare_pca(dataframe)
    make_visualization(PCA_df, labels, plot_title, output_path)


if __name__ == "__main__":
    args, output_title = prepare_script()
    df_test = pd.read_pickle(args.file)

    PCA_df = prepare_pca(df_test)
    make_visualization(PCA_df, LABELS_CIFAR_10, output_title, 
                       f"{args.output}/{output_title}.png")
