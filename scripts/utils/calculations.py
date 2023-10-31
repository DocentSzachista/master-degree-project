import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial import distance

class Distance:
    """Abstract class that for measuring distance"""
    name = "abstract"
    y_lim = (0, 10)
    def count_distance(self, features_origin: np.ndarray, features_finish: np.ndarray):
        raise NotImplementedError("This is abstract method, its not gonna be implemented")


class EuclidianDistance(Distance):
    """Class that implements counting euclidan distance."""

    name = "Euclidian"
    y_lim = (0, 10)

    def count_distance(self, features_origin: np.ndarray, features_finish: np.ndarray):
        """Count euclidean distance.

            :param: origin
            Starting point (original image's features, or final augumentation version)
            :param: target
            Ending point   (point in the space between origin and final version)
        """
        return norm(features_origin - features_finish)


class CosineDistance(Distance):
    """Class that implements counting cosine distance."""

    name = "Cosine"
    y_lim = (0, 1)

    def count_distance(self, features_origin: np.ndarray, features_finish: np.ndarray):
        """Count cosine distance
            :param: origin
            Starting point (original image's features, or final augumentation version)
            :param: target
            Ending point   (point in the space between origin and final version)
        """
        return float(np.dot(
                features_origin, features_finish.T
                        )/(
                norm(features_origin)*norm(features_finish)))


class MahalanobisDistance:
    """Class to count mahalanobis distance."""

    name = "Mahalanobis"

    def count_distance(self, whole_dataset: np.ndarray, features_i: np.ndarray):

        mean = np.mean(whole_dataset, axis=0)
        cov_matrix = np.cov(whole_dataset.T, rowvar=False)

        cos = distance.mahalanobis(
            features_i, mean, cov_matrix
        )
        print(cos)

        # features = features_i - np.mean(whole_dataset.values)
        # covariance = np.linalg.inv(np.cov(whole_dataset.T))
        # left_part = np.dot(features, covariance)
        # mahalanobis = np.dot(left_part, features.T)
        # print(mahalanobis)
        # return mahalanobis
        # y_T = (features_i - features_mean ).T

# mal = MahalanobisDistance()
# df = pd.read_pickle("./dataframes/cifar_10.pickle")
# vectors = df["features"].apply(lambda vec: vec.tolist())
# vectors = vectors.to_list()


# # mu = np.mean(vectors, axis=0)
# # sigma = np.cov(cectors.T)


# cokolwiek = pd.DataFrame(vectors, columns=list(range(1, 2049)))
# # print(len(vectors[0]))
# print(cokolwiek.head())

# # # for vector in vectors:
# # #     print(len(vector))
# df_2 = pd.read_pickle("./dataframes/id_2353.pickle")
# point = df_2.features.to_list()[0].tolist()
# mal.count_distance(
#     df, point
# )

DISTANCE_FUNCS = [
    # MahalanobisDistance(),
    EuclidianDistance(),
    CosineDistance()
]
