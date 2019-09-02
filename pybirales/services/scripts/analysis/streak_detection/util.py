import time

import numpy as np


def get_clusters(ndx, c_labels):
    # Add cluster labels to the data
    labelled_data = np.append(ndx, np.expand_dims(c_labels, axis=1), axis=1)

    # Cluster mask to remove noise clusters
    de_noised_data = labelled_data[labelled_data[:, 3] > -1]

    de_noised_data = de_noised_data[de_noised_data[:, 3].argsort()]

    # Return the location at which clusters where identified
    cluster_ids = np.unique(de_noised_data[:, 3], return_index=True)

    # Split the data into clusters
    clusters = np.split(de_noised_data, cluster_ids[1])

    # remove empty clusters
    clusters = [x for x in clusters if np.any(x)]

    return clusters


def _td(t1):
    return time.time() - t1


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print '{} finished in {:2.4f} seconds'.format(method.__name__, (te - ts))

        return result

    return timed


def get_line_eq(p1, p2):
    if p1[0] == p2[0]:
        return np.inf, 0.

    if p1[1] == p2[1]:
        return 0., 0.

    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    return m, c


def _partition(data, x1, x2, y1, y2):
    ys = data[:, 0]
    partition_y = data[np.logical_and(ys >= y1, ys <= y2)]
    return partition_y[np.logical_and(partition_y[:, 1] >= x1, partition_y[:, 1] <= x2)]


def __ir2(data, i=None):
    if len(data) >= 10:
        return 0., 0., 1
    # print data
    # line is horizontal
    if len(np.unique(data[:, 0])) == 1:
        return -0.09, -1, -1

    # coords = np.flip(np.swapaxes(data[:, :2] - np.mean(data[:, :2], axis=0), 0, -1), 0)
    b = data[:, :2] - np.mean(data[:, :2], axis=0)
    coords = np.flip(b.T, axis=0)
    eigen_values, eigen_vectors = np.linalg.eig(np.cov(coords))
    sort_indices = np.argsort(eigen_values)[::-1]

    # print('Eigen values', eigen_values[sort_indices[0]], eigen_values[sort_indices[1]] )
    # print('Eigen vectors', eigen_vectors[sort_indices[0]], eigen_vectors[sort_indices[1]] )
    p_v1 = eigen_vectors[sort_indices[0]]

    # primary eigenvector is perfectly horizontal or perfectly vertical
    if p_v1[0] == 0 or p_v1[1] == 0:
        return +1, 0, 0

    # Gradient of 1st eigenvector
    m1 = -1 * p_v1[1] / p_v1[0]

    # Gradient of 1st eigenvector (in degrees)
    m2 = -1 * np.arctan2(p_v1[1], p_v1[0])

    return np.abs(eigen_values[sort_indices[1]] / eigen_values[sort_indices[0]]) * np.sign(m1), np.rad2deg(m2), m1
