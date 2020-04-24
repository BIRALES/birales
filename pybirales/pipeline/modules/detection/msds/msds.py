import logging as log

import numpy as np
from numba import njit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import KDTree
from scipy.stats import linregress
from sklearn import linear_model

from util import __ir2, grad2

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
                    # dm[k] = np.vdot(diff[:2], diff[:2]) ** 0.5
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
        # t1 = time.time()
        distances = dm_grad(X, m, empty_dm)
        # print "dm took {} seconds".format(time.time() - t1)
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
        [(cluster, cluster_id, grad2(cluster), np.mean(cluster[:, 1]), np.mean(cluster[:, 0])) for
         (cluster, cluster_id, ratio, bbox) in
         leaves])

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


def h_cluster(X, distance_thold, min_length=2, i=None):
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
            if root.children > min_length:
                # print len(leaves)
                leaves.append((ndx, bbox))

    return leaves


# # @timeit
# def linear_cluster(leave_pair):
#     # better at handling crossing streaks
#     # print np.shape(leave_pair),
#     leave, cluster_id = leave_pair
#     # print cluster_id
#     (data, bbox) = leave
#     labels, _ = h_cluster(data, 2, min_length=2, i=cluster_id)
#     u, c = np.unique(labels, return_counts=True)
#     min_mask = c > 3
#     u_groups = u[min_mask]
#
#     best_ratio = -0.1
#
#     sorted_groups = u_groups[np.argsort(c[min_mask])]
#     n_data = []
#     for g in sorted_groups:
#         c = data[np.where(labels == g)]
#         ratio, _, m1 = __ir2(c, min_n=20, i=cluster_id)
#
#         if 0. >= ratio >= -0.1:
#             best_ratio = ratio
#             if density_check2(c, 0.5, cluster_id):
#                 n_data.append([c, cluster_id + g * .1, ratio, bbox])
#
#     if best_ratio > -0.1:
#         return np.array([1, n_data], dtype=np.object)
#
#     return np.array([-1, [data, -1 * cluster_id, best_ratio, bbox]], dtype=np.object)
#     # return [[data, -1 * cluster_id, best_ratio, bbox]]

# @profile
def linear_cluster(leave_pair):
    # better at handling crossing streaks
    # print np.shape(leave_pair),
    leave, cluster_id = leave_pair
    # print cluster_id
    (data, bbox) = leave
    labels, _ = h_cluster(data, 2, min_length=2, i=cluster_id)
    u, c = np.unique(labels, return_counts=True)
    min_mask = c > 3
    u_groups = u[min_mask]

    best_ratio = -0.1

    sorted_groups = u_groups[np.argsort(c[min_mask])]
    n_data = []
    for g in sorted_groups:
        c = data[np.where(labels == g)]
        ratio, _, m1 = __ir2(c, min_n=20, i=cluster_id)

        if 0. >= ratio >= -0.1:
            best_ratio = ratio
            if density_check2(c, 0.5, cluster_id):
                n_data.append([c, cluster_id + g * .1, ratio, bbox])

    if best_ratio > -0.1:
        return n_data

    # return np.array([-1, [data, -1 * cluster_id, best_ratio, bbox]], dtype=np.object)
    return [[data, -1 * cluster_id, best_ratio, bbox]]


def linear_clusterf(leave_pair):
    # better at handling crossing streaks
    # no rejections

    leave, cluster_id = leave_pair

    (data, bbox) = leave
    labels, u_groups = h_cluster(data, 3, min_length=5, i=cluster_id)

    n_data = []
    for g in u_groups:
        c = data[np.where(labels == g)]
        ratio, _, _ = __ir2(c, min_n=20, i=cluster_id)

        if 0. >= ratio >= -0.1:
            if density_check2(c, 0.5, cluster_id):
                n_data.append([c, cluster_id, ratio, bbox])

    return n_data


def linear_cluster2(leave_pair):
    # This assumes only one cluster per box
    leave, cluster_id = leave_pair

    (data, bbox) = leave
    labels, _ = h_cluster(data, 2, min_length=2, i=cluster_id)
    u, c = np.unique(labels, return_counts=True)
    min_mask = c > 3
    u_groups = u[min_mask]

    best_ratio = -0.1
    best_data = None

    sorted_groups = u_groups[np.argsort(c[min_mask])]

    for g in sorted_groups:
        c = data[np.where(labels == g)]
        ratio, _, m1 = __ir2(c, min_n=20, i=cluster_id)

        if 0. >= ratio >= best_ratio:
            best_ratio = ratio
            best_data = c

    if best_ratio > -0.1 and density_check2(best_data, 0.5, cluster_id):
        return [best_data, cluster_id, best_ratio, np.array(bbox)]

    return [data, -1 * cluster_id, best_ratio, bbox]


def missing_score(param):
    missing = np.setxor1d(np.arange(min(param), max(param)), param)
    return len(missing) / float(len(param))


def density_check2(data, threshold, cluster_id):
    # param = np.unique(data[:, 1])
    # missing = np.setxor1d(np.arange(min(param), max(param) + 1), param)
    # score = len(missing) / float(len(param))

    return missing_score(data[:, 1]) < threshold


# @timeit
def process_leaves(leaves, pool=False, debug=False):
    pos = []
    rej = []
    leave_pairs = [(l, i) for i, l in enumerate(leaves)]
    if pool:
        for clusters in pool.map(linear_cluster, leave_pairs):
            for c in clusters:
                if c[1] < 0:
                    rej.append(c)
                else:
                    pos.append(c)
    else:
        for l in leave_pairs:
            clusters = linear_cluster(l)
            for c in clusters:
                if c[1] < 0:
                    rej.append(c)
                else:
                    pos.append(c)

        # for l in leave_pairs:
        #     pos.extend([c for c in linear_cluster(l) if c[1] < 0]), []

    return pos, rej


def process_leaves2(leaves, pool=False, output_rej=True, debug=False):
    # pos = []
    # for i, l in izip(count(), leaves):
    #     pos.extend(linear_clusterf(leave_pair=(l, i)))
    #
    # return pos, []
    pos = []
    for i in range(len(leaves)):
        pos += linear_clusterf(leave_pair=(leaves[i], i))

    return pos, []


# @timeit
def estimate_leave_eps(positives):
    bboxes = np.vstack(np.array(positives)[:, 3])
    a = bboxes[:, 0] - bboxes[:, 1]
    b = bboxes[:, 2] - bboxes[:, 3]
    return np.median(np.sqrt(a ** 2 + b ** 2)) * 1.5


def add_group(candidate, group):
    return np.append(candidate, np.expand_dims(np.full(len(candidate), group), axis=1), axis=1)


def validate_clusters_func(labelled_cluster, beam_id=-1):
    """

    :param labelled_cluster:
    :return:
    """

    org_candidate, g = labelled_cluster
    ransac = linear_model.RANSACRegressor()
    ransac.fit(org_candidate[:, 0].reshape(-1, 1), org_candidate[:, 1])
    candidate = org_candidate[ransac.inlier_mask_]

    m, intercept, r_value, p, e = linregress(candidate[:, 1], candidate[:, 0])

    param = candidate[:, 1]
    missing = np.setxor1d(np.arange(min(param), max(param)), param)
    score = len(missing) / float(len(param))

    if score > 1:
        log.debug("Candidate {}, dropped since missing score is not high enough ({:0.3f})".format(g, score))
    elif r_value > -.99:
        log.debug("Candidate {}, dropped since r-value is not high enough ({:0.3f})".format(g, r_value))
    elif p > 0.01:
        log.debug("Candidate {}, dropped since p-value is not low enough ({:0.3f})".format(g, p))
    elif e > 0.05:
        log.info("Candidate {}, dropped since correlation error is greater than 0.01 ({})".format(g, e))
        log.info("Candidate {}, {:0.3f} {:0.3f} {:0.3f} {:0.3f} {}".format(g, score, r_value, p, e, len(candidate)))
    else:
        c = add_group(candidate, g)
        if beam_id > -1:
            return [add_group(c, beam_id)]
        return c

    return []


# @timeit
def validate_clusters(data, beam_id=-1, debug=False):
    labelled_clusters = data[:, 5]
    unique_labels = np.unique(labelled_clusters)
    # candidates = [(np.vstack(data[:, 0][labelled_clusters == g]), g) for g in unique_labels]
    clusters = []
    ransac = linear_model.RANSACRegressor()
    for g in unique_labels:
        org_candidate = np.vstack(data[:, 0][labelled_clusters == g])
        ransac.fit(org_candidate[:, 0].reshape(-1, 1), org_candidate[:, 1])
        candidate = org_candidate[ransac.inlier_mask_]

        m, intercept, r_value, p, e = linregress(candidate[:, 1], candidate[:, 0])

        score = missing_score(candidate[:, 1])

        if score > 0.2:
            log.debug("Candidate {}, dropped since missing score is not high enough ({:0.3f})".format(g, score))
        elif r_value > -.98:
            log.debug("Candidate {}, dropped since r-value is not high enough ({:0.3f})".format(g, r_value))
        elif p > 0.01:
            log.debug("Candidate {}, dropped since p-value is not low enough ({:0.3f})".format(g, p))
        elif e > 0.05:
            log.info("Candidate {}, dropped since standard error is greater than 0.01 ({})".format(g, e))
            log.info("Candidate {}, {:0.3f} {:0.3f} {:0.3f} {:0.3f} {}".format(g, score, r_value, p, e, len(candidate)))
        else:
            clusters.append(add_group(add_group(candidate, g), beam_id))
            log.info("Candidate {} is Valid with: {:0.3f} {:0.3f} {:0.3f} {:0.3f} {}".format(g, score, r_value, p, e,
                                                                                             len(candidate)))

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
