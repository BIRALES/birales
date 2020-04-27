import logging as log

import numpy as np
from numba import njit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import KDTree
from scipy.stats import pearsonr
from sklearn import linear_model

from util import __ir, grad2, missing_score

P_VALUE = 0.05
INF_DIST = 10000.
MIN_CLST_DIST = 3
MIN_MBR_LNGTH = 2
MIN_CHILDREN = 3
MIN_UNQ_SMPLS = 15
MIN_IR = 0.001
MIN_SPAN_THLD = 15
MIN_CORR = -0.99
MIN_UNQ_SMPLS_2 = 30
MAX_MISSING_DATA = 0.4
MIN_R = -2901 * 0.10485 / 9.5367  # (min gradient / (channel-bandwidth)) * sampling_rate
MIN_R *= 1.1  # buffer

SLOPE_ANGLE_RANGE = (MIN_R, 0)


# this does not include SNR in the diff calc
@njit("float64[:](float64[:,:], int32, float64[:], float64, int32)")
def dm(X, m, dm, min_r, cid):
    k = 0
    min_r = MIN_R
    for i in xrange(0, m - 1):
        for j in xrange(i + 1, m):
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

    for i in xrange(0, m - 1):
        for j in xrange(i + 1, m):
            diff = X[i] - X[j] + 1e-6
            v[0] = diff[1] / diff[2]
            dm[k] = 10000

            if MIN_R <= v[0] <= 0:
                v[1] = X[i][0]  # slope of point 1
                v[2] = X[j][0]  # slope of point 2

                if -0.2 <= np.std(v) / np.mean(v) <= 0.2:
                    dm[k] = (diff[1] ** 2 + diff[2] ** 2) ** 0.5
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
    return fcluster(Z, threshold, criterion=criterion)


# @timeit
# @profile
def h_cluster_leaves(leaves, distance_thold):
    X = np.vstack(
        [(cluster, cluster_id, grad2(cluster),
          np.average(cluster[:, 1], weights=cluster[:, 2]),
          np.average(cluster[:, 0], weights=cluster[:, 2])) for
         (cluster, cluster_id, ratio, bbox) in
         leaves])

    # X = np.vstack(
    #     [(cluster, cluster_id, grad2(cluster), np.mean(cluster[:, 1]), np.mean(cluster[:, 0]))
    #      for
    #      (cluster, cluster_id, ratio, bbox) in
    #      leaves])

    if len(leaves) == 1:
        return np.append(X, np.array([[0]]), axis=1)

    cluster_labels = fclusterdata(X[:, 2:].astype(float), distance_thold, min_r=MIN_R,
                                  criterion='distance',
                                  check_grad=True)

    u, i, c = np.unique(cluster_labels, return_counts=True, return_index=True)
    min_labels = 1  # was 2
    min_leaves = 1  # was 1
    unique_groups = u[c >= min_labels].astype(int)

    f_groups = []
    for i in range(len(unique_groups)):
        g = unique_groups[i]
        g_i = np.where(cluster_labels == g)
        leave_ids = X[cluster_labels == g][:, 1].astype(int)
        ul, cl = np.unique(leave_ids, return_counts=True)

        # if we have a single big leaf, we consider this as being correct (useful for high SNR)
        if len(ul) > min_leaves or len(X[g_i][0][0]) > 40 * .9:
            f_groups.extend(g_i[0])
    u_mask = np.array(f_groups).astype(int)
    return np.append(X[u_mask], cluster_labels[u_mask].reshape(-1, 1), axis=1)

    # return filter_groups(cluster_data, cluster_labels)


def h_cluster(X, distance_thold, min_length, i=None):
    cluster_labels = fclusterdata(X, distance_thold, criterion='distance', min_r=-1e9, check_grad=False,
                                  cluster_id=i)

    u, c = np.unique(cluster_labels, return_counts=True)
    unique_groups = u[c > min_length]

    return cluster_labels, unique_groups


@njit("float64[:, :](float64[:,:], int32,  int32,  int32,  int32)")
def _partition(data, x1, x2, y1, y2):
    ys = data[:, 0]
    partition_y = data[np.logical_and(ys >= y1, ys <= y2)]
    return partition_y[np.logical_and(partition_y[:, 1] >= x1, partition_y[:, 1] <= x2)]


# @timeit
def traverse(root, ndx, bbox, min_length=2):
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
            ll = traverse(root.less, partition, (_x1, _x2, _y1, _y2), min_length)

            if root.split_dim == 0:
                _y1, _y2 = root.split, y2
            if root.split_dim == 1:
                _x1, _x2 = root.split, x2

            partition = _partition(ndx, _x1, _x2, _y1, _y2)

            lr = traverse(root.greater, partition, (_x1, _x2, _y1, _y2), min_length)
            # rectangles = rectangles + left
            # rectangles = rectangles + right

            leaves = leaves + ll
            leaves = leaves + lr

        else:
            if root.children > min_length:
                # print len(leaves)
                leaves.append((ndx, bbox))

    return leaves


def linear_cluster(leave_pair):
    # better at handling crossing streaks
    # no rejections

    leave, cluster_id = leave_pair

    (data, bbox) = leave
    labels, u_groups = h_cluster(data, 3, min_length=5, i=cluster_id)

    n_data = []
    for g in u_groups:

        c = data[np.where(labels == g)]

        if missing_score(c[:, 1]) > 0.5:
            continue

        ratio = __ir(c, min_n=10, i=cluster_id)

        if 0. >= ratio >= -0.1:
            n_data.append([c, cluster_id, ratio, bbox])

    return n_data


def density_check2(data, threshold, cluster_id):
    return missing_score(data[:, 1]) < threshold


def process_leaves(leaves, debug=False):
    pos = []
    for i in range(len(leaves)):
        pos += linear_cluster(leave_pair=(leaves[i], i))

    return pos


# @timeit
def estimate_leave_eps(positives):
    bboxes = np.vstack(np.array(positives)[:, 3])
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
    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=2)
    # ransac = linear_model.RANSACRegressor()
    for g in unique_labels:
        org_candidate = np.vstack(data[:, 0][labelled_clusters == g])
        r = ransac.fit(org_candidate[:, 0].reshape(-1, 1), org_candidate[:, 1])
        candidate = org_candidate[r.inlier_mask_]

        score = missing_score(candidate[:, 1])
        if score > 0.1:
            log.debug("Candidate {}, dropped since missing score is not high enough ({:0.3f})".format(g, score))
            continue

        r_value, p = pearsonr(candidate[:, 1], candidate[:, 0])

        if r_value > -.99:
            log.debug("Candidate {}, dropped since r-value is not high enough ({:0.3f})".format(g, r_value))
        elif p > 0.01:
            log.debug("Candidate {}, dropped since p-value is not low enough ({:0.3f})".format(g, p))
        else:
            clusters.append(add_group(add_group(candidate, g), beam_id))
            log.info(
                "Candidate {} is Valid with: {:0.3f} {:0.3f} {:0.3f} {}".format(g, score, r_value, p, len(candidate)))
        #
        # m, intercept, r_value, p, e = linregress(candidate[:, 1], candidate[:, 0])
        # if r_value > -.99:
        #     log.debug("Candidate {}, dropped since r-value is not high enough ({:0.3f})".format(g, r_value))
        # elif p > 0.01:
        #     log.debug("Candidate {}, dropped since p-value is not low enough ({:0.3f})".format(g, p))
        # elif e > 0.05:
        #     log.info("Candidate {}, dropped since standard error is greater than 0.01 ({})".format(g, e))
        #     log.info("Candidate {}, {:0.3f} {:0.3f} {:0.3f} {:0.3f} {}".format(g, score, r_value, p, e, len(candidate)))
        # else:
        #     clusters.append(add_group(add_group(candidate, g), beam_id))
        #     log.info("Candidate {} is Valid with: {:0.3f} {:0.3f} {:0.3f} {:0.3f} {}".format(g, score, r_value, p, e,
        #                                                                                      len(candidate)))

    log.debug('Validation reduced {} to {}'.format(len(unique_labels), len(clusters)))

    return clusters


# @timeit
def pre_process_data(test_image, noise_estimate=None):
    if noise_estimate:
        test_image -= noise_estimate

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
