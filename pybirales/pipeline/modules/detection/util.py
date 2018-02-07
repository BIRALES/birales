import numpy as np
import settings

MIN_CHANNEL = settings.observation.start_center_frequency


def is_linear(score, threshold):
    """

    :param score:
    :param threshold:
    :return:
    """
    return score > threshold


def pd(a, b):
    """
    Calculate the percentage difference between two values

    :param a:
    :param b:
    :return:
    """
    diff = a - b
    mean = np.mean([a, b])
    try:
        return abs(diff / mean)
    except RuntimeWarning:
        return 1.0


def are_similar(cluster1, cluster2, threshold):
    """

    :param cluster1:
    :param cluster2:
    :param threshold:
    :return:
    """

    return pd(cluster1.m, cluster2.m) <= threshold and pd(cluster1.c, cluster2.c) <= threshold


def track_cutoff(m, intercept):
    """
    Determine the time at which the track should end

    :param m:
    :param intercept:
    :return:
    """
    return m * MIN_CHANNEL + intercept
