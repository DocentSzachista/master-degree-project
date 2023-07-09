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
        "--train_file", required=True, help="Location of train pickle dataset."
    )
    parser.add_argument(
        "--test_file", required=False, help="location of test pickle dataset."
    )
    parser.add_argument("-o", "--output", required=True, help="Dir to save generated file")
    args = parser.parse_args()
    output_title = re.findall(r"\w+\.", args.train_file)

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


def prepare_pca(dataframe_train: pd.DataFrame,
                dataframe_test: pd.DataFrame | None = None) -> pd.DataFrame:
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
    return final_df


def run(train_input_dataframe_path: str, plot_title: str,
        test_input_dataframe_path: str | None = None,
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
    train_dataframe = pd.read_pickle(train_input_dataframe_path)
    if test_input_dataframe_path is None:
        PCA_df = prepare_pca(train_dataframe)
    else:
        test_df = pd.read_pickle(test_input_dataframe_path)
        PCA_DF = prepare_pca(train_dataframe, test_df)
    make_visualization(PCA_df, labels, plot_title, output_path)


if __name__ == "__main__":
    args, output_title = prepare_script()
    df_train = pd.read_pickle(args.train_file)
    if args.test_file is not None: 
        df_test = pd.read_pickle(args.test_file)
    else: 
        df_test = df_train

    PCA_df = prepare_pca(df_train, df_test)
    make_visualization(PCA_df, LABELS_CIFAR_10, output_title, 
                       f"{args.output}/{output_title}.png")
