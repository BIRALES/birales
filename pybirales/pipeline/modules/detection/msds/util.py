import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


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
        print('{} finished in {:2.4f} seconds'.format(method.__name__, (te - ts)))

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


def eu(x1, x2, y1, y2):
    diff = np.array([x1, y1]) - np.array([x2, y2])

    return np.vdot(diff, diff) ** 0.5


def __ir2(data, min_n=10, i=None):
    if len(data) >= min_n:
        return -0.09123, -1, -1

    # line is horizontal
    if len(np.unique(data[:, 0])) == 1:
        return -0.0913, -1, -1

    b = data[:, :2] - np.mean(data[:, :2], axis=0)
    coords = np.flip(b.T, axis=0)
    eigen_values, eigen_vectors = np.linalg.eig(np.cov(coords))
    sort_indices = np.argsort(eigen_values)[::-1]

    p_v1 = eigen_vectors[sort_indices[0]]

    # primary eigenvector is perfectly horizontal or perfectly vertical
    if p_v1[0] == 0 or p_v1[1] == 0:
        return +1, 0, 0

    # Gradient of 1st eigenvector
    m1 = -1 * p_v1[1] / p_v1[0]

    # Gradient of 1st eigenvector (in degrees)
    m2 = -1 * np.arctan2(p_v1[1], p_v1[0])

    ir = np.abs(eigen_values[sort_indices[1]] / eigen_values[sort_indices[0]]) * np.sign(m1)

    # radius = eu(data[:, 1].min(), data[:, 1].max(), data[:, 0].min(), data[:, 0].max())  / 2
    # ir_normalised = ir / (len(data) * radius** 2)

    # print ir, ir_normalised
    return ir, np.rad2deg(m2), m1


def __ir(data, min_n=10, i=None):
    if len(data) >= min_n:
        return -0.09123

    # line is horizontal
    if len(np.unique(data[:, 0])) == 1:
        return -0.09123

    b = data[:, :2] - np.mean(data[:, :2], axis=0)
    coords = np.flip(b.T, axis=0)
    eigen_values, eigen_vectors = np.linalg.eig(np.cov(coords))
    sort_indices = np.argsort(eigen_values)[::-1]

    p_v1 = eigen_vectors[sort_indices[0]]

    # primary eigenvector is perfectly horizontal or perfectly vertical
    if p_v1[0] == 0 or p_v1[1] == 0:
        return 1

    # Gradient of 1st eigenvector
    m1 = -1 * p_v1[1] / p_v1[0]

    return np.abs(eigen_values[sort_indices[1]] / eigen_values[sort_indices[0]]) * np.sign(m1)


def _create_cluster(cluster_data, channels, obs_info, beam_id, iter_count):
    """

    :param cluster_data: The labelled cluster data
    :param channels:
    :param t0:
    :param td:
    :param beam_id:
    :param iter_count:
    :return:
    """
    t0 = np.datetime64(obs_info['timestamp'])
    td = np.timedelta64(int(obs_info['sampling_time'] * 1e9), 'ns')
    channel_ndx = cluster_data[:, 0].astype(int)
    time_ndx = cluster_data[:, 1].astype(int)
    snr_data = cluster_data[:, 2]
    time_sample = time_ndx + iter_count * 160

    # Calculate the channel (frequency) of the sample from the channel index
    channel = channels[channel_ndx]

    # Calculate the time of the sample from the time index
    time = t0 + (time_sample - 160 * iter_count) * td

    return pd.DataFrame({
        'time_sample': time_sample,
        'channel_sample': channel_ndx,
        'time': time,
        'channel': channel,
        'snr': snr_data,
        'beam_id': np.full(time_ndx.shape[0], beam_id),
        'iter': np.full(time_ndx.shape[0], iter_count),
    })


def grad2(cluster, ir=None):
    if ir == -0.09123:
        return -0.09123

    _, i = np.unique(cluster[:, 1], return_index=True)
    cluster = cluster[i]

    _, i = np.unique(cluster[:, 0], return_index=True)
    cluster = cluster[i]

    if len(cluster) == 1:
        return -0.09123

    cov = np.cov(cluster[:, 1], cluster[:, 0]).flatten()

    return cov[0] / (cov[1] + 1e-9)


def snr_calc(track, noise_estimate):
    diff = (track[:, 2] - noise_estimate)
    diff[diff <= 0] = np.nan

    track[:, 2] = 10 * np.log10(diff / noise_estimate)

    return track


def _validate_clusters(clusters):
    return [c for c in clusters if is_valid(c, r_thold=-0.9, min_span_thold=1)]


def is_valid(cluster, r_thold=-0.98, min_span_thold=1, group=-1):
    c = cluster[:, 0]
    s = cluster[:, 1]

    # ur = len(np.unique(c)) * 1.0 / (max(c) - min(c))
    # if ur < 0.4:
    #     return False
    #
    # if len(np.unique(s)) == 1 or len(np.unique(c)) == 1:
    #     return False
    #
    # if len(np.unique(s)) < min_span_thold:
    #     return False

    pear = pearsonr(s, c)

    if pear[1] < 0.05 and pear[0] < r_thold:
        return True

    return False


def missing_score(param):
    missing = np.setxor1d(np.arange(min(param), max(param)), param)
    return len(missing) / float(len(param))
