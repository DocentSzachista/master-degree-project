import numpy as np
from numpy.linalg import norm

def count_euclidian_distance(features_origin: np.ndarray, features_finish: np.ndarray):
    """Count euclidean distance.

        :param: origin
        Starting point (original image's features, or final augumentation version)
        :param: target
        Ending point   (point in the space between origin and final version)
    """
    return norm(features_origin - features_finish)


def count_cosine_distance(features_origin: np.ndarray, features_finish: np.ndarray):
    """Count cosine distance
        :param: origin
        Starting point (original image's features, or final augumentation version)
        :param: target
        Ending point   (point in the space between origin and final version)
    """
    return np.dot(
        features_origin,features_finish
                  )/(
        norm(features_origin)*norm(features_finish))
