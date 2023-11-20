from scripts.plots import confussion
from scripts.setup import Setup, Worker
from scripts.utils.calculations import MahalanobisDistance
import pandas as pd 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.colors as mcolors


if __name__ == "__main__":
    setup = Setup()
    # confussion.run(setup.config, [50, 100, 150, 200, 250, 300, 500], f"./visualizations/test-numbers-mixup")

    dist = MahalanobisDistance()
    path = "./resnet-noise_all_dataset/noise/dataframes/0/id_3.pickle"
    path_2 = "./datasets/cifar_10.pickle"
    path_3 = "./datasets/test_cifar10.pickle"
    df = pd.read_pickle(path)
    train_df = pd.read_pickle(path_2)
    test_df = pd.read_pickle(path_3)
    
    print(test_df.columns)
    print(train_df.columns)
    dist.fit(train_df)
    res = dist.count_distance(df)
    # print(res)

    # print((train_df.features.values))
    # print((np.stack(train_df.features.to_numpy())))

    pca = PCA(n_components=2)
    pca.fit(np.stack(train_df.features.to_numpy()))
    test_pca = pca.transform(np.stack(test_df.features.to_numpy()))
    data_pca = pca.transform(np.stack(df.features.to_numpy()))
    colors = mcolors.TABLEAU_COLORS.keys()

    df = pd.DataFrame(test_pca, columns=["PC1", "PC2"])
    df = pd.concat((df, test_df.original_label ), axis=1)
    plt.figure(figsize=(10, 6))


    for color, label in zip(colors, range(0, 9)):
        indices_to_keep = df["original_label"] == label
        plt.scatter(
            df.loc[indices_to_keep, "PC1"],
            df.loc[indices_to_keep, "PC2"],
            label=label,
            s=50,
            c=color,
        )


    # plt.scatter(test_pca[:, 0], test_pca[:, 1], c=res, cmap='viridis', edgecolor='k')
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=res, cmap='viridis', edgecolor='k', marker="v")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Mahalanobis Distance with PCA')
    plt.show()