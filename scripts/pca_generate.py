import argparse
from os import makedirs, path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import re
from sample_picker import pick_chosen_label
import matplotlib.colors as mcolors

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
        "--train_file", required=True, help="Location of train pickle dataset."
    )
    parser.add_argument(
        "--test_file", required=False, help="location of test pickle dataset."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Dir to save generated file"
    )
    args = parser.parse_args()
    output_title = re.findall(r"\w+\.", args.train_file)

    return args, output_title[0]


def generate_pca_scatter_plot(
    df: pd.DataFrame,
    labels: dict,
    plot_title: str,
) -> None:
    """Visualize PCA

    Draw model's class distribution on a scatter plot.

    Parameters
    ----------
    df: DataFrame
        Dataframe containing PCA and original labels keys
    labels: dict
        Labels for each of possible classifier values.

    Returns
    -------
    Figure and Axes objects.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("PCA 1", fontsize=15)
    ax.set_ylabel("PCA 2", fontsize=15)
    ax.set_title(plot_title, fontsize=20)
    colors = mcolors.TABLEAU_COLORS.keys()
    for color, label in zip(colors, labels.keys()):
        indices_to_keep = df["original_label"] == label
        ax.scatter(
            df.loc[indices_to_keep, "PC1"],
            df.loc[indices_to_keep, "PC2"],
            label=labels[label],
            s=50,
            c=color
        )
    ax.legend()
    ax.grid()
    return fig, ax


def mark_chosen_datapoints(
    axis: plt.Axes, pca: PCA, points_dataframe: pd.DataFrame
) -> plt.Axes:
    """Mark datapoints on already existing Axes object.

    Parameters:
    -----------
    axis: Axes,
        Axis where additional points will be mapped.
    pca: PCA
        Pretrained PCA object which will reduce dimensions of provided points.
    points_dataframe: Dataframe
        Dataframe containing points to be marked on scatter plot.

    Returns:
    --------
    Modified Axes object.
    """
    features = np.vstack(points_dataframe.features.values)
    reduced_X = pca.transform(features)
    df = pd.DataFrame(reduced_X, columns=["PC1", "PC2"])
    axis.scatter(df.loc[:, "PC1"], df.loc[:, "PC2"], s=100, marker="x", c="k")
    return axis


def prepare_pca(
    dataframe_train: pd.DataFrame, dataframe_test: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Prepare dataset for PCA visualization.

    Make dimension reduction to 2D by using PCA algorithm.

    Parameters
    -----
    dataframe_train: DataFrame
        Dataframe, that holds features and labels of deep learning model used to train PCA.
    dataframe_test: Dataframe, optional
        Dataframe holding test features, if not provided, function uses train data to transform features.
    Returns
    -------
    Dataframe containing original labels and reduced dimensions of features.

    """
    pca = PCA(n_components=2)
    train_features = np.vstack(dataframe_train.features.values)
    if dataframe_test is not None:
        test_features = np.vstack(dataframe_test.features.values)
        labels = dataframe_test.original_label
    else:
        test_features = train_features
        labels = dataframe_train.original_label
    pca.fit(train_features)
    X_reduced = pca.transform(test_features)
    df = pd.DataFrame(X_reduced, columns=["PC1", "PC2"])
    final_df = pd.concat([df, labels], axis=1)
    return final_df, pca


def run(
    train_input_dataframe_path: str,
    plot_title: str,
    test_input_dataframe_path: str | None = None,
    picked_datapoints: pd.DataFrame | None = None,
    labels: dict | None = LABELS_CIFAR_10,
    output_path: str | None = None,
) -> None:
    """Util function to run PCA generation.

    Parameters
    ----------
    input_dataframe_path: str
        Path to train pickle file.
    plot_title: str
        Name of generated plot
    test_input_dataframe_path: str, optional
        Path to test pickle file.
    picked_datapoints: DataFrame, optional
        Dataframe containing picked points to show their exact location on PCA diagram.
    labels: dict, optional
        Labels describing keys in the dataframe. If not provided it will use CIFAR_10 labels.
    output_file: str, optional
        Directory to the file where it should save plot.
    """
    train_dataframe = pd.read_pickle(train_input_dataframe_path)
    if test_input_dataframe_path is None:
        PCA_df = prepare_pca(train_dataframe)
    else:
        test_df = pd.read_pickle(test_input_dataframe_path)
        PCA_df = prepare_pca(train_dataframe, test_df)
    fig, axis = generate_pca_scatter_plot(PCA_df, labels, plot_title)
    if picked_datapoints is not None:
        mark_chosen_datapoints(axis, PCA_df, picked_datapoints)
    if output_path is None:
        plt.show()
    else: 
        plt.savefig(output_path)


if __name__ == "__main__":
    args, output_title = prepare_script()
    df_train = pd.read_pickle(args.train_file)
    if args.test_file is not None:
        df_test = pd.read_pickle(args.test_file)
    else:
        df_test = df_train

    PCA_df, pca = prepare_pca(df_train, df_test)
    # make_visualization(PCA_df, LABELS_CIFAR_10, output_title,
    #                    f"{args.output}/{output_title}.png")
    (_, axis) = generate_pca_scatter_plot(PCA_df, LABELS_CIFAR_10, output_title)

    points = pick_chosen_label(df_test, 4).head(5)
    points.to_csv("./cokolwiek.csv")
    mark_chosen_datapoints(axis, pca, points.head(5))
    plt.savefig(f"{args.output}/{output_title}_test_pca.png")

