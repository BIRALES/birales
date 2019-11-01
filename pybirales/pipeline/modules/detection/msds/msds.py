# from multiprocessing import Pool

import multiprocessing

import numpy as np
from numba import njit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import KDTree
from scipy.stats import pearsonr
from sklearn import linear_model
from scipy.stats import linregress
from util import timeit, __ir2, grad, grad2, grad3
from visualisation import visualise_ir2
from sklearn.linear_model import LinearRegression

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
# MIN_R = -10
SLOPE_ANGLE_RANGE = (MIN_R, 0)


# # @profile
# @njit(["float64(float64[:],float64[:])"], fastmath=False)
# def mydist2(p1, p2):
#     diff = p1 - p2 + 1e-6
#
#     if (diff[0] / diff[1]) > 0:
#         return 10000.
#
#     return np.vdot(diff, diff) ** 0.5
#
#
# @njit(["float64(float64[:],float64[:])"], fastmath=False)
# def mydist3(p1, p2):
#     diff = p1 - p2 + 1e-6  # dx = p1[1] - p2[1],  dy = p1[0] - p2[0]
#
#     # if -0.987 <= diff[1] / diff[0] <= -0.0056:
#     if -1 <= diff[1] / diff[0] <= 0:
#         return np.vdot(diff, diff) ** 0.5
#     return 10000.
#

def h_cluster(X, distance_thold, min_length=2, i=None):
    cluster_labels = fclusterdata_denis(X, distance_thold, criterion='distance', min_r=-1e9, check_grad=False,
                                        cluster_id=i)

    u, c = np.unique(cluster_labels, return_counts=True)
    unique_groups = u[c > min_length]

    return cluster_labels, unique_groups


@njit
def pdist_denis2(X, m, dm, min_r):
    k = 0
    min_r = MIN_R
    for i in xrange(0, m - 1):
        for j in xrange(i + 1, m):
            diff = X[i] - X[j]

            # perfectly horizontal
            if diff[0] == 0 and abs(diff[1]) > 1:
                dm[k] = 100
            # vertical
            elif diff[1] == 0 and abs(diff[0]) > 1:
                dm[k] = 100
            else:
                diff += 1e-6
                slope = diff[0] / diff[1]
                if min_r <= slope <= 0:
                    eu = np.vdot(diff, diff) ** 0.5
                    dm[k] = eu

                    # if perfectly vertical, only consider point if it is very near
                    # if slope < -100 and abs(diff[0]) > 1:
                    #     dm[k] = 10000

                # dm[k] = np.vdot(diff, diff) ** 0.5
            k = k + 1
    return dm


@njit
def pdist_denis_w_grad(X, m, dm, min_r):
    k = 0
    v = np.zeros(shape=3)
    for i in xrange(0, m - 1):
        for j in xrange(i + 1, m):
            diff = X[i] - X[j] + 1e-6
            v[0] = diff[1] / diff[2]
            if min_r <= v[0] <= 0:
                v[1] = X[i][0]  # gradient of point 1
                v[2] = X[j][0]  # slope of point 2
                if -0.2 <= np.std(v) / np.mean(v) <= 0.2:
                    dm[k] = np.vdot(diff[1:], diff[1:]) ** 0.5
            k = k + 1
    return dm


# @njit
# def pdist_denis_w_grad(X, m, dm, min_r):
#     k = 0
#     v = np.zeros(shape=3)
#     for i in xrange(0, m - 1):
#         for j in xrange(i + 1, m):
#             diff = X[i] - X[j] + 1e-6
#             v[0] = diff[2] / diff[1]
#
#             if min_r <= v[0] <= 0:
#                 dm[k] = np.vdot(diff[1:], diff[1:]) ** 0.5
#             k = k + 1
#     return dm


# @profile
def fclusterdata_denis(X, threshold, criterion, min_r=-1e9, check_grad=False, cluster_id=0):
    # Compute the distance matrix
    # distances = pdist(X, metric)

    m = X.shape[0]
    dm = np.full((m * (m - 1)) // 2, 10000, dtype=np.float)

    if check_grad:
        distances = pdist_denis_w_grad(X, m, dm, min_r)
    else:
        distances = pdist_denis2(X, m, dm, min_r)

    # Perform single linkage clustering
    linked = linkage(distances, 'single')

    # auto-selection of threshold based on density
    # t = 2 * min(linked[:, 2])
    # if min_r <= -10:
    #     threshold = t

    # Determine to which cluster each initial point would belong given a distance threshold
    return fcluster(linked, threshold, criterion=criterion)


@timeit
def h_cluster_euclidean(X, distance_thold):
    cluster_labels = fclusterdata_denis(X[:, 2:5].astype(float), distance_thold, min_r=MIN_R, criterion='distance',
                                        check_grad=True)
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


def linear_cluster(leave, distance_thold=3, cluster_id=0):
    (data, bbox) = leave
    labels, _ = h_cluster(data, 3, min_length=2, i=cluster_id)
    u, c = np.unique(labels, return_counts=True)
    min_mask = c > 3
    u_groups = u[min_mask]

    best_ratio = -0.1
    best_data = data
    # start with the smallest grouping
    x1, x2, y1, y2 = bbox

    sorted_groups = u_groups[np.argsort(c[min_mask])]
    for g in sorted_groups:
        c = data[np.where(labels == g)]
        ratio, _, m1 = __ir2(c, min_n=20, i=cluster_id)

        if cluster_id == 5294:
            visualise_ir2(c, data, cluster_id, 0)
            print ratio, grad(c), len(c), grad2(c), grad3(c), m1

        # if 0. >= ratio >= best_ratio and SLOPE_ANGLE_RANGE[0] < grad(c) < SLOPE_ANGLE_RANGE[1]:
        if 0. >= ratio >= best_ratio:
            best_ratio = ratio
            best_data = c

    slope = grad(best_data)

    mc, ms, msnr = np.mean(best_data, axis=0)
    if best_ratio > -0.1:
        return [best_data, 0, best_ratio, x1, x2, y1, y2, best_data.shape[0], ms, mc, msnr, slope, cluster_id]

    best_ratio = '{:0.3f}\n{:0.3f}'.format(best_ratio, slope)
    return [data, -1, best_ratio, x1, x2, y1, y2, best_data.shape[0], ms, mc, msnr, slope, cluster_id]


@timeit
def process_leaves(pool, leaves, parallel=True):
    pos = []
    rej = []

    if parallel:
        for c in pool.map(linear_cluster, leaves):
            if c[1] == -1:
                rej.append(c)
            else:
                pos.append(c)
    else:
        for i, l in enumerate(leaves):
            c = linear_cluster(l, cluster_id=i)
            if c[1] == -1:
                rej.append(c)
            else:
                pos.append(c)

    return pos, rej


def estimate_leave_eps(clusters):
    subset = np.array(clusters)[:, 3:7].astype(float)
    diff = subset[:, [0, 2]] - subset[:, [1, 3]]
    mean_leave_size = np.median(diff, axis=0)

    return np.vdot(mean_leave_size, mean_leave_size) ** 0.5 + 1 * np.std(diff)


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

    # sample_gaps = np.arange(0, 160)
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


def fill2(test_image, cluster, fill_thickness=False):
    """

    :param test_image:
    :param cluster:
    :param fill_thickness:
    :return:
    """

    x = cluster[:, 1]
    y = cluster[:, 0]

    group = cluster[0][3]

    # Get the missing samples (x)
    x_range = np.arange(min(x), max(x))
    x_gaps = np.setxor1d(x_range, x)

    # Get the missing channels(y)
    y_range = np.arange(min(y), max(y))
    y_gaps = np.setxor1d(y_range, y)

    x_seg = [s for s in np.array_split(x_gaps, np.where(np.diff(x_gaps) > 3)[0] + 1) if len(s) < 15]

    model = LinearRegression()
    # linear_model.RANSACRegressor(residual_threshold=5)
    model.fit(x.reshape(-1, 1), y)

    if len(x_seg) < 1:
        return cluster

    X = np.concatenate(x_seg)
    y_predict = model.predict(X=X.reshape(-1, 1)).astype(int)

    combined = np.zeros(shape=(len(y_predict), 4), dtype=int)
    combined[:, 0] = y_predict
    combined[:, 1] = X
    combined[:, 2] = test_image[combined[:, 0], combined[:, 1]]

    # Remove duplicate rows with the same value and channel
    # Select sample with highest power per time sample
    s_combined = combined[combined[:, 2].argsort()[::-1]]
    u, i = np.unique(s_combined[:, 1], return_index=True)
    s_combined = s_combined[i]

    # Combine the filled-in data with the original cluster
    filled_cluster = np.concatenate([cluster, s_combined])

    ransac = linear_model.RANSACRegressor()
    ransac.fit(filled_cluster[:, 0].reshape(-1, 1), filled_cluster[:, 1])

    cluster = filled_cluster[ransac.inlier_mask_]
    combined = cluster
    # Remove duplicate rows with the same value and channel
    # Select sample with highest power per time sample
    s_combined = combined[combined[:, 2].argsort()[::-1]]
    u, i = np.unique(s_combined[:, 1], return_index=True)
    s_combined = s_combined[i]

    return s_combined

    # t = np.mean(s_combined[:, 2]) - np.std(s_combined[:, 2])*2
    #
    # return s_combined[s_combined[:, 2] > t]

    # if np.sum(~ransac.inlier_mask_) > 0.2 * len(candidate):

    combinations = np.array(np.meshgrid(y_gaps, x_gaps), dtype=int).T.reshape(-1, 2)

    combined = np.zeros(shape=(len(combinations), 4))
    combined[:, 0:2] = combinations

    # Add the power of the channels, samples
    combined[:, 2] = test_image[combinations[:, 0], combinations[:, 1]]

    # Remove duplicate rows with the same value and channel
    # Select sample with highest power per time sample
    s_combined = combined[combined[:, 2].argsort()[::-1]]
    u, i = np.unique(s_combined[:, 1], return_index=True)
    s_combined = s_combined[i]

    # Combine the filled-in data with the original cluster
    filled_cluster = np.concatenate([cluster, s_combined])

    t = np.median(cluster[:, 2]) - np.std(cluster[:, 2]) * 2

    print 'Cluster {}; was filled with {} new data points'.format(cluster[0][3], len(cluster) - len(
        filled_cluster[filled_cluster[:, 2] > t]))
    # return filled_cluster
    return filled_cluster[filled_cluster[:, 2] > t]


def add_group(candidate, group):
    return np.append(candidate, np.expand_dims(np.full(len(candidate), group), axis=1), axis=1)


# @profile
def split(ransac, candidate):
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


def validate_clusters_func(labelled_cluster):
    tracks = []
    candidate, g = labelled_cluster
    ransac = linear_model.RANSACRegressor(residual_threshold=5)

    snr = candidate[:, 2]

    # filter false positives - based on their SNR values
    t = np.median(snr) - np.std(snr) * 2

    candidate = candidate[snr > t]

    unique_samples, indices = np.unique(candidate[:, 1], return_index=True)

    if len(unique_samples) < MIN_UNQ_SMPLS:
        print 'not enough unique samples', len(unique_samples), g, np.mean(candidate[:, 0])
        return tracks

    # skip validation if IR is very low
    if abs(__ir2(candidate, min_n=len(candidate) + 1)[0]) < MIN_R:
        print 'Ir small enough', np.mean(candidate[:, 0]), g
        return [add_group(candidate, g)]

        # return tracks

    ransac.fit(candidate[:, 0].reshape(-1, 1), candidate[:, 1])
    candidate = candidate[ransac.inlier_mask_]

    valid = is_valid(candidate, min_span_thold=MIN_SPAN_THLD, group=g)

    if g == 22:
        print 'is_valid', valid

    tmp = [candidate]
    if not valid:
        tmp = split(ransac, candidate)

    for t_track in tmp:

        t_track = density_check(t_track)

        if not np.any(t_track):
            print 'not dense', g
            continue

        s = t_track[:, 1]
        c = t_track[:, 0]

        if np.corrcoef(s, c)[0][1] < MIN_CORR or len(np.unique(s)) > MIN_UNQ_SMPLS_2:
            candidate = add_group(t_track, g)
            tracks.append(candidate)
        else:
            print 'Not corr or not long enough', g, np.corrcoef(s, c)[0][1], len(np.unique(s))

    return tracks


def validate_clusters_func2(labelled_cluster):
    tracks = []
    candidate, g = labelled_cluster
    ransac = linear_model.RANSACRegressor(residual_threshold=5)

    snr = candidate[:, 2]

    ransac.fit(candidate[:, 0].reshape(-1, 1), candidate[:, 1])

    c = pearsonr(candidate[:, 0], candidate[:, 1])



    print 'Group G: R:{:0.3f}, E:{:0.3f}, P:{:0.3f}, IR:{:0.3f}'.format(

    __ir2(candidate, min_n=len(candidate) + 1)[0])

    # filter false positives - based on their SNR values
    t = np.median(snr) - np.std(snr) * 2

    candidate = candidate[snr > t]

    unique_samples, indices = np.unique(candidate[:, 1], return_index=True)

    if len(unique_samples) < MIN_UNQ_SMPLS:
        print 'not enough unique samples', len(unique_samples), g, np.mean(candidate[:, 0])
        return tracks

    # skip validation if IR is very low
    if abs(__ir2(candidate, min_n=len(candidate) + 1)[0]) < MIN_R:
        print 'Ir small enough', np.mean(candidate[:, 0]), g
        return [add_group(candidate, g)]

        # return tracks

    ransac.fit(candidate[:, 0].reshape(-1, 1), candidate[:, 1])
    candidate = candidate[ransac.inlier_mask_]

    valid = is_valid(candidate, min_span_thold=MIN_SPAN_THLD, group=g)

    if g == 22:
        print 'is_valid', valid

    tmp = [candidate]
    if not valid:
        tmp = split(ransac, candidate)

    for t_track in tmp:

        t_track = density_check(t_track)

        if not np.any(t_track):
            print 'not dense', g
            continue

        s = t_track[:, 1]
        c = t_track[:, 0]

        if np.corrcoef(s, c)[0][1] < MIN_CORR or len(np.unique(s)) > MIN_UNQ_SMPLS_2:
            candidate = add_group(t_track, g)
            tracks.append(candidate)
        else:
            print 'Not corr or not long enough', g, np.corrcoef(s, c)[0][1], len(np.unique(s))

    return tracks



def density_check(t_track):
    channels = t_track[:, 0]
    samples = t_track[:, 1]

    missing_channels = np.setxor1d(np.arange(min(channels), max(channels)), channels)
    r_channels = len(missing_channels) / float(len(missing_channels) + len(channels))

    if r_channels >= MAX_MISSING_DATA:
        sorted_track = t_track[(-t_track[:, 1]).argsort()]
        gaps = np.where(np.diff(sorted_track[:, 0]) > 3)[0]
        segments = np.array_split(sorted_track, gaps)

        valid = [seg for seg in segments if len(seg) > 5]

        if valid:
            return np.vstack(valid)

        return False
    missing_samples = np.setxor1d(np.arange(min(samples), max(samples)), samples)
    r_samples = len(missing_samples) / float(len(missing_samples) + len(samples))

    if r_samples >= MAX_MISSING_DATA:
        sorted_track = t_track[(-t_track[:, 1]).argsort()]
        gaps = np.where(np.diff(sorted_track[:, 1]) > 3)[0]
        segments = np.array_split(sorted_track, gaps)
        valid = [seg for seg in segments if len(seg) > 5]
        if valid:
            return np.vstack(valid)

        return False

    return t_track


@timeit
def validate_clusters(pool, data, unique_labels):
    labelled_clusters = data[:, 6]
    candidates = [(np.vstack(data[:, 0][labelled_clusters == g]), g) for g in unique_labels]
    # clusters = [c for sub_clusters in pool.map(validate_clusters_func, candidates) if sub_clusters for c in
    #             sub_clusters]

    clusters = []
    for c in candidates:
        clusters.extend(validate_clusters_func2(c))
    # clusters = [validate_clusters_func(c) for c in candidates]

    print 'Validation reduced {} to {}'.format(len(unique_labels), len(clusters))

    return clusters


def is_valid(cluster, r_thold=-0.9, min_span_thold=1, group=-1):
    c = cluster[:, 0]
    s = cluster[:, 1]

    ur = len(np.unique(c)) * 1.0 / (max(c) - min(c))
    if ur < 0.4:
        print 'uniq < 0.4', group
        return False
    # if len(cluster) < MIN_UNQ_SMPLS:
    #     return False

    if len(np.unique(s)) == 1 or len(np.unique(c)) == 1:
        print 'uniq = 1', group
        return False

    if len(np.unique(s)) < min_span_thold:
        print 'uniq min than span', group, len(np.unique(s))
        return False

    pear = pearsonr(s, c)

    if pear[1] < P_VALUE and pear[0] < r_thold:
        return True
    else:
        print 'pear[1] < P_VALUE and pear[0] < r_thold', group, pear

    return False


@timeit
def cluster_leaves(leaves, distance_thold):
    cluster_data = np.vstack(
        [(cluster, j, m, x, y, p) for i, (cluster, best_gs, ratio, x1, x2, y1, y2, n, x, y, p, m, j) in
         enumerate(leaves)])

    if len(leaves) == 1:
        return np.append(cluster_data, np.full(fill_value=1, shape=cluster_data.shape[0]).reshape(-1, 1), axis=1), [1]

    cluster_labels, unique_labels = h_cluster_euclidean(cluster_data, distance_thold)
    # print cluster_data.shape, cluster_labels.shape, unique_labels
    return np.append(cluster_data, cluster_labels.reshape(-1, 1), axis=1), unique_labels


@timeit
def pre_process_data(test_image, noise_estimate):
    # print np.min(test_image), noise_estimate, np.shape(test_image)
    ndx = np.column_stack(np.where(test_image > 0.))
    # print 'ndx', ndx.shape, test_image.shape
    # print len(ndx), min(ndx[:, 0]), max(ndx[:, 0])

    power = test_image[test_image > 0.]

    snr = 10 * np.log10(power / noise_estimate)
    ndx = np.append(ndx, np.expand_dims(snr, axis=1), axis=1)
    # ndx = np.append(ndx, np.expand_dims(power, axis=1), axis=1)
    return ndx


def pre_process_data2(test_image):
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
