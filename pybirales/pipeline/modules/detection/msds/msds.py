import logging as log

import numpy as np
from numba import njit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import KDTree
from scipy.stats import pearsonr
from sklearn import linear_model

from pybirales.pipeline.modules.detection.msds.util import __ir, grad2, missing_score

MIN_CLST_DIST = 3
MIN_CHILDREN = 3
MIN_IR = 0.001
# MIN_R = -2901 * 0.10485 / 9.5367  # (min gradient / (channel-bandwidth)) * sampling_rate
# MIN_R *= 1.1  # buffer

# MIN_R = -291 * 0.10485 / 9.5367  # (min gradient / (channel-bandwidth)) * sampling_rate
# MIN_R *= 1.1  # buffer
#
# MAX_R = 1.1 * -57 * 0.10485 / 9.5367

# sampling_time =  9.5367 # seconds
# channel_bandwidth = 0.10485 # Hz

sampling_time = 0.0958698057142857  # seconds
channel_bandwidth = 10.43081283569336  # Hz

MIN_R = -291 * sampling_time / channel_bandwidth  # (min gradient / (channel-bandwidth)) * sampling_rate
MIN_R *= 1.1  # buffer

MAX_R = 1.1 * -57 * sampling_time / channel_bandwidth


# this does not include SNR in the diff calc
@njit("float64[:](float64[:,:], int32, float64[:], float64, int32)")
def dm(X, m, dm, min_r, cid):
    k = 0
    min_r = MIN_R
    # min_r = -35e3
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            diff = X[i] - X[j]

            # perfectly horizontal and distance is large  (> 1 px)
            if diff[0] == 0 and abs(diff[1]) > 1:
                dm[k] = 10000
            # vertical
            elif diff[1] == 0 and abs(diff[0]) > 1:
                dm[k] = 10000
            else:
                diff += 1e-6
                if min_r <= (diff[0] / diff[1]) <= 0:
                    dm[k] = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
            k = k + 1
    return dm


@njit("float64[:](float64[:,:], int32, float64[:])")
def dm_grad(X, m, dm):
    k = 0
    v = np.zeros(shape=3)

    for i in range(0, m - 1):
        for j in range(i + 1, m):
            diff = X[i] - X[j] + 1e-6
            v[0] = diff[1] / diff[2]
            dm[k] = 10000

            if MIN_R <= v[0] <= 0:
                v[1] = X[i][0]  # slope of point 1
                v[2] = X[j][0]  # slope of point 2

                if v[1] == -0.09123 or v[2] == -0.09123:
                    dm[k] = (diff[1] ** 2 + diff[2] ** 2) ** 0.5
                elif -0.3 <= np.std(v) / np.mean(v) <= 0.3:
                    dm[k] = (diff[1] ** 2 + diff[2] ** 2) ** 0.5

                # if -0.3 <= np.std(v) / np.mean(v) <= 0.3:
                #     dm[k] = (diff[1] ** 2 + diff[2] ** 2) ** 0.5
            k = k + 1
    return dm


# @profile
def fclusterdata(X, threshold, criterion, min_r=-1e9, check_grad=False, cluster_id=0):
    m = X.shape[0]
    empty_dm = np.full((m * (m - 1)) // 2, 10000, dtype=np.float)

    if check_grad:
        # Compute the distance matrix
        distances = dm_grad(X, m, empty_dm)
    else:
        distances = dm(X, m, empty_dm, min_r, cluster_id)

    # Perform single linkage clustering
    Z = linkage(distances, 'single')

    # Determine to which cluster each initial point would belong given a distance threshold
    labels = fcluster(Z, threshold, criterion=criterion)

    # plt.scatter(X[:, 1], X[:, 0], c=labels)

    return labels


# @timeit
# @profile
def h_cluster_leaves(leaves, distance_thold):
    a = [(cluster, cluster_id, grad2(cluster, ratio),
          np.average(cluster[:, 1], weights=cluster[:, 2]),
          np.average(cluster[:, 0], weights=cluster[:, 2])) for
         (cluster, cluster_id, ratio, bbox) in
         leaves]
    X = np.vstack(np.array(a, dtype=object))

    if len(leaves) == 1:
        return np.append(X, np.array([[0]]), axis=1)

    cluster_labels = fclusterdata(X[:, 2:].astype(float), distance_thold, min_r=MIN_R,
                                  criterion='distance',
                                  check_grad=True)

    u, i, c = np.unique(cluster_labels, return_counts=True, return_index=True)
    min_labels = 2  # was 2
    unique_groups = u[c >= min_labels].astype(int)

    f_groups = []
    for i in range(len(unique_groups)):
        g = unique_groups[i]
        g_i = np.where(cluster_labels == g)

        f_groups.extend(g_i[0])

    u_mask = np.array(f_groups).astype(int)
    return np.append(X[u_mask], cluster_labels[u_mask].reshape(-1, 1), axis=1)


def h_cluster(X, distance_thold, min_length, i=None):
    cluster_labels = fclusterdata(X, distance_thold, criterion='distance', check_grad=False,
                                  cluster_id=i)

    u, c = np.unique(cluster_labels, return_counts=True)
    unique_groups = u[c >= min_length]

    return cluster_labels, unique_groups


@njit("float64[:, :](float64[:,:], int32,  int32,  int32,  int32)")
def _partition(data, x1, x2, y1, y2):
    ys = data[:, 0]
    partition_y = data[np.logical_and(ys >= y1, ys <= y2)]
    return partition_y[np.logical_and(partition_y[:, 1] >= x1, partition_y[:, 1] <= x2)]


# @timeit
def traverse(root, ndx, bbox, noise_est, min_length=2):
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
            ll = traverse(root.less, partition, (_x1, _x2, _y1, _y2), noise_est, min_length)

            if root.split_dim == 0:
                _y1, _y2 = root.split, y2
            if root.split_dim == 1:
                _x1, _x2 = root.split, x2

            partition = _partition(ndx, _x1, _x2, _y1, _y2)

            lr = traverse(root.greater, partition, (_x1, _x2, _y1, _y2), noise_est, min_length)
            # rectangles = rectangles + left
            # rectangles = rectangles + right

            leaves = leaves + ll
            leaves = leaves + lr

        else:
            if root.children > min_length:
                # print len(leaves)

                p5 = noise_est * 10 ** (5 / 10.)  # power at SNR = 5
                pm = np.mean(ndx[:, 2])  # mean power of leaf

                ndx = ndx[ndx[:, 2] > (pm - p5)]

                if np.any(ndx):
                    leaves.append((ndx, bbox))

    return leaves


def linear_cluster(leave_pair):
    (data, bbox), leaf_id = leave_pair

    min_length = 5

    if len(data) < 2:
        return [[data, leaf_id, -0.09321, bbox]]

    labels, u_groups = h_cluster(data, 4, min_length=min_length, i=leaf_id)

    n_data = []
    for g in u_groups:
        c = data[np.where(labels == g)]
        if missing_score(c[:, 1]) > 0.5 and missing_score(c[:, 0]) > 0.5:
            continue

        ratio = __ir(c, min_n=20, i=leaf_id)

        if 0. >= ratio >= -0.15:
            n_data.append([c, leaf_id, ratio, bbox])

    return n_data


def process_leaves(leaves, debug=False):
    pos = []
    for i in range(len(leaves)):
        pos += linear_cluster(leave_pair=(leaves[i], i))

    return pos


# @timeit
def estimate_leave_eps(positives):
    bboxes = np.vstack(np.array(positives, dtype=object)[:, 3])
    a = bboxes[:, 0] - bboxes[:, 1]
    b = bboxes[:, 2] - bboxes[:, 3]
    return np.median(np.sqrt(a ** 2 + b ** 2)) * 1.5


def add_group(candidate, group):
    return np.append(candidate, np.expand_dims(np.full(len(candidate), group), axis=1), axis=1)


# @timeit
def validate_clusters(data, beam_id=-1, debug=False):
    labelled_clusters = data[:, 5]
    unique_labels = np.unique(labelled_clusters)
    clusters = []

    # random seed for ransac is so that results are consistent

    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),
                                          residual_threshold=2, random_state=500)

    for g in unique_labels:
        org_candidate = np.vstack(data[:, 0][labelled_clusters == g])
        r = ransac.fit(org_candidate[:, 0].reshape(-1, 1), org_candidate[:, 1])

        c1 = is_valid(org_candidate[r.inlier_mask_], g, beam_id)
        if np.any(c1):
            clusters.append(c1)

            c2 = org_candidate[~r.inlier_mask_]
            if len(c2) > 3 and missing_score(c2[:, 1]) < 2:
                r = ransac.fit(c2[:, 0].reshape(-1, 1), c2[:, 1])
                c2 = c2[r.inlier_mask_]
                c2 = is_valid(c2, g * -1, beam_id)

                if np.any(c2):
                    clusters.append(c2)

    log.debug('Validation reduced {} to {}'.format(len(unique_labels), len(clusters)))

    return clusters


def is_valid(candidate, g, beam_id):
    if len(candidate) < 1:
        return False

    c_score = missing_score(candidate[:, 1])
    t_score = missing_score(candidate[:, 0])
    if c_score > 0.5 and t_score > 0.5:
        log.debug(f"Candidate {g}, dropped since missing score is high (c={c_score:0.3f}, t={t_score:0.3f})")
        return False

    r_value, p = pearsonr(candidate[:, 1], candidate[:, 0])

    if r_value > -.99:
        log.debug("Candidate {}, dropped since r-value is not high enough ({:0.3f})".format(g, r_value))
        return False
    elif p > 0.01:
        log.debug("Candidate {}, dropped since p-value is not low enough ({:0.3f})".format(g, p))
        return False

    log.debug(
        "Candidate {} is Valid with: cs={:0.3f} ts={:0.3f} r={:0.3f} p={} n={}".format(g, c_score, t_score, r_value, p,
                                                                                       len(candidate)))

    return add_group(add_group(candidate, g), beam_id)


# @timeit
def pre_process_data(test_image):
    ndx = np.column_stack(np.where(test_image > 0.))
    power = test_image[test_image > 0.]

    return np.append(ndx, np.expand_dims(power, axis=1), axis=1)


# @timeit
def build_tree(ndx, leave_size, n_axis):
    """
    Build a 2D or 3D kd tree.

    At a high SNR, a 3D kd tree is slow

    :param ndx:
    :param leave_size:
    :param n_axis:
    :return:
    """

    return KDTree(ndx[:, :n_axis], leave_size)
