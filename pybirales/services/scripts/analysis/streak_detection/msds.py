# from multiprocessing import Pool

import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import KDTree
from scipy.stats import pearsonr
from sklearn import linear_model

from util import timeit, __ir2
from visualisation import visualise_post_processed_tracks, visualise_tracks, visualise_clusters, \
    visualise_filtered_data, visualise_tree_traversal


# @profile
@njit(["float64(float64[:],float64[:])"], fastmath=False)
def mydist2(p1, p2):
    diff = p1 - p2 + 1e-6

    if (diff[0] / diff[1]) > 0:
        return 10000.

    return np.vdot(diff, diff) ** 0.5


@njit(["float64(float64[:],float64[:])"], fastmath=False)
def mydist3(p1, p2):
    diff = p1 - p2 + 1e-6  # dx = p1[1] - p2[1],  dy = p1[0] - p2[0]

    # if -0.987 <= diff[1] / diff[0] <= -0.0056:
    if -1 <= diff[1] / diff[0] <= 0:
        return np.vdot(diff, diff) ** 0.5
    return 10000.


def h_cluster(X, distance_thold, min_length=2, i=None):
    # Y = preprocessing.scale(X)
    # a = preprocessing.normalize(X)

    # scaler = MinMaxScaler()
    # sx = scaler.fit(X).transform(X)
    # r = np.max(X, axis=0) - np.min(X, axis=0)
    # cluster_labels = fclusterdata_denis(sx, 0.3, criterion='distance')

    cluster_labels = fclusterdata_denis(X, 3, criterion='distance')

    u, c = np.unique(cluster_labels, return_counts=True)
    unique_groups = u[c > 2]

    return cluster_labels, unique_groups


@njit
def pdist_denis2(X, m, dm, min_r):
    k = 0
    for i in xrange(0, m - 1):
        for j in xrange(i + 1, m):
            diff = X[i] - X[j] + 1e-6
            if min_r <= (diff[1] / diff[0]) <= 0:
                dm[k] = np.vdot(diff, diff) ** 0.5
            k = k + 1
    return dm


# @profile
def fclusterdata_denis(X, threshold, criterion, min_r=-1e9):
    # Compute the distance matrix
    # distances = pdist(X, metric)

    m = X.shape[0]
    dm = np.full((m * (m - 1)) // 2, 10000, dtype=np.float)
    distances = pdist_denis2(X, m, dm, min_r)

    # Perform single linkage clustering
    linked = linkage(distances, 'single')

    # auto-selection of threshold based on density
    # t = 2 * min(linked[:, 2])
    # if min_r <= -10:
    #     threshold = t

    # Determine to which cluster each initial point would belong given a distance threshold
    return fcluster(linked, threshold, criterion=criterion)


def h_cluster_euclidean(X, distance_thold):
    cluster_labels = fclusterdata_denis(X[:, 2:4].astype(float), distance_thold, min_r=-1, criterion='distance')

    # cluster_labels = fclusterdata(X[:, 2:4].astype(float), distance_thold, metric=mydist3, criterion='distance')

    u, c = np.unique(cluster_labels, return_counts=True)
    unique_groups = u[c > 1]
    return cluster_labels, unique_groups


# @njit
def _partition(data, x1, x2, y1, y2):
    ys = data[:, 0]
    partition_y = data[np.logical_and(ys >= y1, ys <= y2)]
    return partition_y[np.logical_and(partition_y[:, 1] >= x1, partition_y[:, 1] <= x2)]


# @profile
# @timeit
def traverse(root, ndx, bbox, distance_thold=3.0, min_length=2, cluster_size_thold=10):
    """

    :param root: The tree node
    :param ndx:
    :param bbox: The bounding box of the partition
    :param distance_thold:
    :param min_length:
    :param cluster_size_thold:
    :return:
    """
    # rectangles = []
    leaves = []
    x1, x2, y1, y2 = bbox
    if root:
        if not isinstance(root, KDTree.leafnode):
            # rectangles.append((root.split_dim, root.split, x1, x2, y1, y2, root.children))

            _x1, _x2, _y1, _y2 = bbox
            if root.split_dim == 0:
                _y1, _y2 = y1, root.split
            if root.split_dim == 1:
                _x1, _x2 = x1, root.split

            partition = _partition(ndx, _x1, _x2, _y1, _y2)
            ll = traverse(root.less, partition, (_x1, _x2, _y1, _y2), distance_thold, min_length,
                                cluster_size_thold)

            if root.split_dim == 0:
                _y1, _y2 = root.split, y2
            if root.split_dim == 1:
                _x1, _x2 = root.split, x2

            partition = _partition(ndx, _x1, _x2, _y1, _y2)

            lr = traverse(root.greater, partition, (_x1, _x2, _y1, _y2), distance_thold, min_length,
                                 cluster_size_thold)
            # rectangles = rectangles + left
            # rectangles = rectangles + right

            leaves = leaves + ll
            leaves = leaves + lr

        else:
            if root.children > 3:
                leaves.append((ndx, bbox))

    return leaves


def eu(x1, x2, y1, y2):
    diff = np.array([x1, y1]) - np.array([x2, y2])

    return np.vdot(diff, diff) ** 0.5


def linear_cluster(fclust1, bb, bbox=None, i=None):
    u, c = np.unique(fclust1, return_counts=True)
    min_mask = c > 3
    u_groups = u[min_mask]

    best_ratio = -0.1
    best_data = bb
    # start with the smallest grouping
    sorted_groups = u_groups[np.argsort(c[min_mask])]
    for g in sorted_groups:
        c = bb[np.where(fclust1 == g)]
        ratio, _, _ = __ir2(c, i)

        if 0. >= ratio >= best_ratio:
            best_ratio = ratio
            best_data = c

    if best_ratio > -0.1:
        mc, ms, msnr = np.mean(best_data, axis=0)
        x1, x2, y1, y2 = bbox
        return [best_data, 0, best_ratio, x1, x2, y1, y2, best_data.shape[0], ms, mc, msnr, i]

    return None


def pl(leave, distance_thold=3):
    (data, bbox) = leave
    labels, _ = h_cluster(data, distance_thold, min_length=2, i=0)
    return linear_cluster(labels, data, bbox=bbox, i=0)


@timeit
def process_leaves2(leaves):
    pool = multiprocessing.Pool(processes=8)

    clusters = [c for c in pool.map(pl, leaves) if c]
    pool.close()

    return clusters


def estimate_leave_eps(clusters):
    subset = np.array(clusters)[:, 3:7].astype(float)
    diff = subset[:, [0, 2]] - subset[:, [1, 3]]
    mean_leave_size = np.mean(diff, axis=0)

    return np.vdot(mean_leave_size, mean_leave_size) ** 0.5 + np.std(diff)


def estimate_thickness(cluster, axis=0):
    _, c = np.unique(cluster[:, axis], return_counts=True)
    return np.median(c)


def thicken(sample, channel, slope, c, offset_sample, offset_channel):
    na = len(sample)
    nb = len(channel)

    a = int(offset_sample - 0.5 * offset_sample)
    b = int(offset_channel - 0.5 * offset_channel)

    offsets_a = np.arange(-a, a)
    offsets_b = np.arange(-b, b)

    offsets_a = offsets_a[offsets_a != 0]
    offsets_b = offsets_b[offsets_b != 0]

    nna = na * (len(offsets_a))
    filled = np.zeros(shape=(nna + nb * (len(offsets_b)), 2))
    if a > 0:
        for i, o in enumerate(offsets_a):
            filled[i * na:i * na + na, :] = np.column_stack([channel, sample + o]).astype(int)

    if b > 0:
        for i, o in enumerate(offsets_b):
            filled[nna + (i * nb):nna + (i * nb + nb), :] = np.column_stack([channel + o, sample]).astype(int)

    return filled.astype(int)


def fill(test_image, cluster, fill_thickness=False):
    ransac = linear_model.RANSACRegressor(residual_threshold=5)
    ransac = ransac.fit(cluster[:, 0].reshape(-1, 1), cluster[:, 1])
    sample = cluster[:, 1]
    channel = cluster[:, 0]
    missing_channels = np.setxor1d(np.arange(min(channel), max(channel)), channel)
    n = len(missing_channels)
    # if n > len(cluster):
    #     return cluster

    sample_gaps = np.arange(min(sample), max(sample))
    missing_samples = np.setxor1d(sample_gaps, sample)

    predicted_channel = ((missing_samples - ransac.estimator_.intercept_) / ransac.estimator_.coef_[0]).astype(int)

    predicted_samples = ransac.predict(missing_channels.reshape(-1, 1))

    m = predicted_samples < test_image.shape[1]
    predicted_samples = predicted_samples[m]
    missing_channels = missing_channels[m]

    new_samples = np.concatenate([predicted_samples, missing_samples]).astype(int)
    new_channels = np.concatenate([missing_channels, predicted_channel]).astype(int)

    # Add missing values functionality
    filled_missing = np.column_stack(
        [new_channels, new_samples, test_image[new_channels, new_samples]])
    filled_missing = np.append(filled_missing, np.expand_dims(np.full(len(filled_missing), -2), axis=1), axis=1)

    # Thickening functionality (axis should be inverse)
    if fill_thickness:
        channel_width = estimate_thickness(cluster, axis=1)
        sample_width = estimate_thickness(cluster, axis=0)
        filled_in_thick = thicken(new_samples, new_channels, ransac.estimator_.coef_[0],
                                  ransac.estimator_.intercept_,
                                  sample_width,
                                  channel_width)
        power = test_image[filled_in_thick[:, 0], filled_in_thick[:, 1]]
        filled_in_thick = np.append(filled_in_thick, np.expand_dims(power, axis=1), axis=1)
        filled_in_thick = np.append(filled_in_thick, np.expand_dims(np.full(len(filled_in_thick), -3), axis=1),
                                    axis=1)
        filled_missing = np.concatenate([filled_missing, filled_in_thick])

    # Combine the extracted data with the predicted
    combined = np.concatenate([cluster, filled_missing])

    # Get rows that have a unique value for channel and row
    u, i = np.unique(combined[:, :2], axis=0, return_index=True)

    # Remove duplicate rows with the same value and channel
    combined = combined[i]

    # filter false positives - based on their SNR values
    t = np.median(cluster[:, 2]) - np.std(cluster[:, 2]) * 2

    # print t, np.mean(test_image[test_image > 0]) * 0.5
    return combined[combined[:, 2] > t]


def add_group(candidate, group):
    return np.append(candidate, np.expand_dims(np.full(len(candidate), group), axis=1), axis=1)


# @profile
def split2(ransac, candidate):
    candidates = []
    ransac.fit(candidate[:, 0].reshape(-1, 1), candidate[:, 1])
    if np.sum(~ransac.inlier_mask_) > 0.2 * len(candidate):
        split_candidate_1 = candidate[ransac.inlier_mask_]
        split_candidate_2 = candidate[~ransac.inlier_mask_]

        ransac.fit(split_candidate_1[:, 0].reshape(-1, 1), split_candidate_1[:, 1])
        split_candidate_1 = split_candidate_1[ransac.inlier_mask_]

        if is_valid(split_candidate_1):
            candidates.append(split_candidate_1)

        ransac.fit(split_candidate_2[:, 0].reshape(-1, 1), split_candidate_2[:, 1])
        split_candidate_2 = split_candidate_2[ransac.inlier_mask_]

        if is_valid(split_candidate_2):
            candidates.append(split_candidate_2)

    return candidates

def validate_clusters_func(cluster):
    ransac = linear_model.RANSACRegressor(residual_threshold=5)
    validated_clusters = validate_cluster(ransac, cluster, 0)
    return validated_clusters

@timeit
def validate_clusters3(data, labelled_clusters, unique_labels):
    pool = multiprocessing.Pool(processes=4)

    candidates = [np.vstack(data[:, 0][labelled_clusters == g]) for g in unique_labels]
    clusters = [c for sub_clusters in pool.map(validate_clusters_func, candidates) if sub_clusters for c in sub_clusters]
    pool.close()

    return clusters

@timeit
# @profile
def validate_clusters2(data, labelled_clusters, unique_labels, true_tracks, limits, visualisation=False):
    ransac = linear_model.RANSACRegressor(residual_threshold=5)
    tracks = []

    candidates = [np.vstack(data[:, 0][labelled_clusters == g]) for g in unique_labels]
    for g, candidate in enumerate(candidates):
        validated_clusters = validate_cluster(ransac, candidate, g)

        if validated_clusters:
            tracks.extend(validated_clusters)

    print 'Reduced', len(candidates) - len(tracks), 'candidates to', len(tracks), 'tracks in validation'

    # visualise_tracks(tracks, true_tracks, '5_tracks.png', limits=limits, debug=visualisation)

    return tracks


def is_valid(cluster, r_thold=-0.9, min_length=10, min_span_thold=1):
    c = cluster[:, 0]
    s = cluster[:, 1]
    if len(cluster) < min_length:
        return False

    if len(np.unique(s)) == 1 or len(np.unique(c)) == 1:
        return False

    if len(np.unique(s)) < min_span_thold:
        return False

    pear = pearsonr(s, c)

    if pear[1] < 0.05 and pear[0] < r_thold:
        return True

    return False


# # filter false positives - based on their SNR values
#     t = np.median(cluster[:, 2]) - np.std(cluster[:, 2]) * 2
#
#     # print t, np.mean(test_image[test_image > 0]) * 0.5
#     return combined[combined[:, 2] > t]

# @profile
def validate_cluster(ransac, candidate, g):
    tracks = []

    snr = candidate[:, 2]
    s = candidate[:, 1]

    # filter false positives - based on their SNR values
    t = np.median(snr) - np.std(snr) * 2

    candidate = candidate[snr > t]

    ransac.fit(candidate[:, 0].reshape(-1, 1), candidate[:, 1])
    candidate = candidate[ransac.inlier_mask_]

    if len(np.unique(s)) < 15:
        return tracks

    valid = is_valid(candidate, min_span_thold=15)

    tmp = [candidate]
    if not valid:
        tmp = split2(ransac, candidate)

    for t_track in tmp:
        s = t_track[:, 1]
        c = t_track[:, 0]

        if len(np.unique(s)) < 15:
            continue

        missing_channels = np.setxor1d(np.arange(min(c), max(c)), c)

        r = len(missing_channels) / float(len(missing_channels) + len(c))
        if r >= 0.40:
            continue

        if np.corrcoef(s, c)[0][1] < -0.99:
            candidate = add_group(t_track, g)
            tracks.append(candidate)

    return tracks


@timeit
def fill_clusters2(tracks, test_img, true_tracks, visualisation=False):
    filled_tracks = [fill(test_img, t) for t in tracks]

    visualise_post_processed_tracks(filled_tracks, true_tracks, '6_post-processed-tracks.png', limits=None,
                                    debug=visualisation)

    print 'Last validation removed', len(tracks) - len(filled_tracks), 'tracks from', len(tracks)

    return filled_tracks


@timeit
def cluster_leaves(leaves, distance_thold, true_tracks, limits, visualisation=False):
    cluster_data = np.vstack(
        [(cluster, j, x, y, p) for i, (cluster, best_gs, ratio, x1, x2, y1, y2, n, x, y, p, j) in enumerate(leaves)])

    cluster_labels, unique_labels = h_cluster_euclidean(cluster_data, distance_thold)

    visualise_clusters(cluster_data, cluster_labels, unique_labels, true_tracks, filename='3_clusters.png',
                       limits=limits,
                       debug=visualisation)

    return cluster_labels, unique_labels, cluster_data


@timeit
def pre_process_data(test_image, noise_estimate):
    ndx = np.column_stack(np.where(test_image > 0.))

    power = test_image[test_image > 0.]

    snr = 10 * np.log10(power / noise_estimate)
    ndx = np.append(ndx, np.expand_dims(snr, axis=1), axis=1)
    # ndx = np.append(ndx, np.expand_dims(power, axis=1), axis=1)
    return ndx


@timeit
def build_tree(ndx, leave_size, n_axis, true_tracks, limits, debug=False, visualisation=False):
    """
    Build a 2D or 3D kd tree.

    At a high SNR, a 3D kd tree is slow

    :param ndx:
    :param leave_size:
    :param n_axis:
    :return:
    """

    visualise_filtered_data(ndx, true_tracks, '1_filtered_data', limits=limits, debug=debug, vis=visualisation)

    return KDTree(ndx[:, :n_axis], leave_size)


def visualise_ir2(data, group, ratio2):
    if len(data) >= 10:
        return 0., 0., data

    ms, mc = np.mean(data[:, 1]), np.mean(data[:, 0])
    coords = np.flip(np.swapaxes(data[:, :2] - np.mean(data[:, :2], axis=0), 0, -1), 0)
    eigen_values, eigen_vectors = np.linalg.eig(np.cov(coords))
    sort_indices = np.argsort(eigen_values)[::-1]

    x_v1, y_v1 = eigen_vectors[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = eigen_vectors[:, sort_indices[1]]  # Eigenvector with the second largest eigenvalue

    scale = 6

    ax = plt.axes()
    plt.plot([x_v1 * -scale * 1 + ms, x_v1 * scale * 1 + ms],
             [y_v1 * -scale * 1 + mc, y_v1 * scale * 1 + mc], color='black')
    plt.plot([x_v2 * -scale + ms, x_v2 * scale + ms],
             [y_v2 * -scale + mc, y_v2 * scale + mc], color='blue')

    plt.plot(data[:, 1], data[:, 0], '.', markersize=12, zorder=1)
    ratio, _, _ = __ir2(data)
    print ratio, len(data)
    ax.text(0.05, 0.95,
            'Group: {}\nRatio: {:0.5f}'.format(group, ratio), color='k',
            weight='bold',
            fontsize=15,
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)

    # line_1 = (x1, y1), (x2, y2)
    # line_2 = (x12, y12), (x22, y22)

    # print data[membership(line_1, line_2, data, membership_ratio=0.7), :].shape

    plt.show()
