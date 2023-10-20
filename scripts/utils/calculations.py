import numpy as np
from numpy.linalg import norm
import pandas as pd


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

    def count_distance(self, whole_dataset: pd.Series, features_i: np.ndarray):

        features = features_i - np.mean(whole_dataset.values)
        covariance = np.linalg.inv(np.cov(whole_dataset.T))
        left_part = np.dot(features, covariance)
        mahalanobis = np.dot(left_part, features.T)
        print(mahalanobis)
        return mahalanobis
        # y_T = (features_i - features_mean ).T


DISTANCE_FUNCS = [
    # MahalanobisDistance(),
    EuclidianDistance(),
    CosineDistance()
]
