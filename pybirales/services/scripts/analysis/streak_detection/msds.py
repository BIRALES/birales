import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import KDTree
from scipy.stats import linregress, pearsonr
from sklearn import linear_model
import matplotlib.pyplot as plt
from util import timeit, get_line_eq


# @profile
@njit(fastmath=True)
def mydist2(p1, p2):
    diff = p1[:3] - p2[:3]  # dx = p1[1] - p2[1],  dy = p1[0] - p2[0]
    # print p1, p2
    if diff[1]:  # if dx and dy are not 0.
        if np.arctan(diff[0] / diff[1]) > 0:
            return 10000.

    return np.vdot(diff, diff) ** 0.5


def h_cluster(X, distance_thold, min_length=2):
    cluster_labels = fclusterdata(X, distance_thold, metric=mydist2, criterion='distance')

    u, c = np.unique(cluster_labels, return_counts=True)
    unique_groups = u[c > min_length]

    return cluster_labels, unique_groups


def h_cluster_euclidean(X, distance_thold):
    cluster_labels = fclusterdata(X[:, 2:4], distance_thold, metric='euclidean', criterion='distance')

    u, c = np.unique(cluster_labels, return_counts=True)
    unique_groups = u[c > 1]
    return cluster_labels, unique_groups


# @njit
def __ir2(data, i=0, g=0):
    coords = np.flip(np.swapaxes(data - np.mean(data, axis=0), 0, -1), 0)
    cov = np.cov(coords)

    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]

    if x_v1 == 0. or x_v2 == 0.:
        return 1

    p1 = np.array([-x_v1, y_v2])
    p2 = np.array([x_v1, y_v2])
    diff1 = p1 - p2
    pa1 = np.vdot(diff1, diff1) ** 0.5 + 0.0000000001

    p1 = np.array([-x_v2, y_v1])
    p2 = np.array([x_v2, -y_v1])
    diff2 = p1 - p2
    pa2 = np.vdot(diff2, diff2) ** 0.5 + 0.0000000001

    m1 = y_v1 / x_v1
    m2 = y_v2 / x_v2

    d1, d2 = nearest(m1, m2, coords[0, :], coords[1, :])

    pa1n = np.sum(d1 < d2) + 1.
    pa2n = np.sum(d1 >= d2) + 1.

    return pa2n / pa1n * pa2 / pa1


def eigen_vectors(data):
    coords = np.flip(np.swapaxes(data[:, :2] - np.mean(data[:, :2], axis=0), 0, -1), 0)

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]  # Eigenvector with the second largest eigenvalue

    return x_v1, y_v1, x_v2, y_v2


def __ir(data, i=0, g=0):
    x_v1, y_v1, x_v2, y_v2 = eigen_vectors(data)

    if x_v1 == 0. or x_v2 == 0.:
        return 1

    diff1 = np.array([-2 * x_v1, 0])
    pa1 = np.vdot(diff1, diff1) ** 0.5

    diff2 = np.array([-2 * x_v2, 2 * y_v1])
    pa2 = np.vdot(diff2, diff2) ** 0.5

    ev_1, ev_2 = ((-x_v1, -y_v1), (x_v1, y_v1)), ((-x_v2, -y_v2), (x_v2, y_v2))
    d1 = dist_to_line(data, ev_1)

    d2 = dist_to_line(data, ev_2)

    # d1, d2 = nearest(m1, m2, coords[0, :], coords[1, :])

    pa1n = np.sum(d1 < d2) + 1.
    pa2n = np.sum(d1 >= d2) + 1.

    return pa2n / pa1n * pa2 / pa1

    return pa2 / pa1


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
    rectangles = []
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
            left, ll = traverse(root.less, partition, (_x1, _x2, _y1, _y2), distance_thold, min_length,
                                cluster_size_thold)

            if root.split_dim == 0:
                _y1, _y2 = root.split, y2
            if root.split_dim == 1:
                _x1, _x2 = root.split, x2

            partition = _partition(ndx, _x1, _x2, _y1, _y2)

            right, lr = traverse(root.greater, partition, (_x1, _x2, _y1, _y2), distance_thold, min_length,
                                 cluster_size_thold)
            rectangles = rectangles + left
            rectangles = rectangles + right

            leaves = leaves + ll
            leaves = leaves + lr

        else:
            if root.children > 3:
                labels, _ = h_cluster(ndx, distance_thold)
                ratio, best_gs, cluster, rejected_data = __best_group(labels, ndx, bbox=bbox)
                n = len(cluster)

                mesg = 'B:{}\n N:{}\nI:{:0.3f}'.format(best_gs, n, ratio)

                if best_gs > 0:
                    pear = pearsonr(cluster[:, 1], cluster[:, 0])

                    if n >= cluster_size_thold or pear[1] <= 0.05:
                        leaves.append(
                            (cluster, best_gs, mesg, x1, x2, y1, y2, n, np.mean([x1, x2]), np.mean([y1, y2]),
                             np.mean(cluster[:, 2])))
                    else:
                        mesg += '\nS:{:0.3f}'.format(pear[1])
                        rectangles.append((cluster, rejected_data, best_gs, mesg, x1, x2, y1, y2, n))
                else:
                    mesg += '\nG:{}'.format(best_gs)
                    rectangles.append((cluster, rejected_data, best_gs, mesg, x1, x2, y1, y2, n))

    return rectangles, leaves


# @profile
def __best_group(fclust1, bb, bbox=None, i=None):
    u, c = np.unique(fclust1, return_counts=True)
    u_groups = u[c > 2]

    best_ratio = 0.2
    best_group = None
    best_data = None
    rejected_data = []
    x1, x2, y1, y2 = bbox
    # bbox_area = (x2 - x1) * (y2 - y1)
    for j, g in enumerate(u_groups):
        c = bb[np.where(fclust1 == g)]
        ratio, c2 = __inertia_ratio2(c, j, g)
        # _, _ = __inertia_ratio2(c, j, g)
        # print
        # cluster_area = (max(c2[:, 0]) - min(c2[:, 0])) * (max(c2[:, 1]) - min(c2[:, 1]))
        # thres = len(bb) / bbox_area * cluster_area
        if ratio < best_ratio:
            best_ratio = ratio
            best_group = g
            best_data = c2
        else:
            rejected_data.append((c2, ratio))

    if not best_group:
        return best_ratio, -1, bb, rejected_data
    # todo - merge best groups

    return best_ratio, best_group, best_data, rejected_data


# @profile
# @njit
def nearest4(m1, m2, x, y, i=None, membership_ratio=0.7):
    d1, d2 = nearest(m1, m2, x, y)
    t = d1 + d2 + 0.000000000001

    mask = d1 < d2

    a = 1. - (d2 / t)
    a[mask] = (1. - (d1 / t))[mask]

    # Select those points that are near to either of the two line by 70%
    return a > membership_ratio


def nearest2(x0, y0, x1, y1, x2, y2):
    num = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    return num / denom


def dist_to_line(point, line):
    vertex_1, vertex_2 = line
    x0, y0 = point[:, 1] - np.mean(point[:, 1]), point[:, 0] - np.mean(point[:, 0])
    x1, y1 = vertex_1
    x2, y2 = vertex_2
    num = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    return num / denom


def membership(line_1, line_2, data, membership_ratio=0.7):
    d_line1 = dist_to_line(data, line_1)

    d_line2 = dist_to_line(data, line_2)

    t = (d_line1 + d_line2) + 0.000000000001

    mask = d_line1 < d_line2

    a = 1. - (d_line2 / t)
    a[mask] = (1. - (d_line1 / t))[mask]

    # Select those points that are near to either of the two line by 70%
    return a > membership_ratio


def nearest(m1, m2, x, y, i=None):
    if m1 == 0:
        line1_x = x
        line1_y = 0.
    else:
        line1_x = y / m1
        line1_y = m1 * x

    if m2 == 0.:
        line2_x = x
        line2_y = 0.
    else:
        line2_x = y / m2
        line2_y = m2 * x

    return np.hypot(line1_y - y, line1_x - x), np.hypot(line2_y - y, line2_x - x)


# @profile
# @njit
def __inertia_ratio(data, j=0, g=0, visualise=None):
    n = data.shape[0]

    if n >= 10:
        return 0.19, data

    if np.unique(data[:, 1]).shape[0] < 2 or np.unique(data[:, 0]).shape[0] < 2:
        return 0.21, data

    coords = np.flip(np.swapaxes(data[:, :2] - np.mean(data[:, :2], axis=0), 0, -1), 0)

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]  # Eigenvector with the second largest eigenvalue

    if x_v1 == 0. or x_v2 == 0.:
        return 0.21, data

    m1 = y_v1 / x_v1

    if m1 > 0:
        return 20., data

    m2 = y_v2 / x_v2
    s = nearest4(m1, m2, coords[0, :], coords[1, :], membership_ratio=0.7)

    subset = data[s, :]
    print j, len(subset)
    if len(subset) < 3:
        return 30., data

    ratio2 = __ir(subset[:, :2])

    # print g, j, n
    # if g == 7 and j == 1 and n == 9:
    #     print m1, m2
    #     visualise_ir(data, g, ratio2)

    return ratio2, subset


def __inertia_ratio2(data, j=0, g=0, visualise=None):
    n = data.shape[0]

    if n >= 10:
        return 0.19, data

    if np.unique(data[:, 1]).shape[0] < 2 or np.unique(data[:, 0]).shape[0] < 2:
        return 0.21, data

    x_v1, y_v1, x_v2, y_v2 = eigen_vectors(data)

    ev_1, ev_2 = ((-x_v1, -y_v1), (x_v1, y_v1)), ((-x_v2, -y_v2), (x_v2, y_v2))

    if x_v1 == 0. or x_v2 == 0.:
        return 20, data

    # non negative slope
    if y_v1 / x_v1 > 0:
        return 20., data

    m = membership(ev_1, ev_2, data, membership_ratio=0.7)

    subset = data[m, :]

    if len(subset) < 3:
        return 30., data

    ratio2 = __ir(subset[:, :2])

    # if g == 7 and j == 1 and n == 9:
    #     print j, len(subset)
    #     visualise_ir(data, g, ratio2)

    return ratio2, subset


def estimate_thickness(cluster, axis=0):
    _, c = np.unique(cluster[:, axis], return_counts=True)
    return np.median(c)


def predict_y(x, slope, c, offset):
    return np.column_stack([slope * x + (c + offset), x])


def predict_x(y, slope, c, offset):
    return np.column_stack([(y - (c + offset)) / slope, y])


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


def fit(cluster, in_liers, r_thold=-0.9, min_length=10):
    d = cluster[in_liers]

    if len(np.unique(d[:, 1])) == 1 or len(np.unique(d[:, 0])) == 1:
        return cluster, False

    pear = pearsonr(d[:, 1], d[:, 0])

    if pear[1] < 0.05 and pear[0] < r_thold and len(d) >= min_length:
        return d, True

    return cluster, False


def similar(candidate, cluster2, i=None, j=None):
    c1_m = np.array([np.mean(candidate[:, 0]), np.mean(candidate[:, 1])])
    c2_m = np.array([np.mean(cluster2[:, 0]), np.mean(cluster2[:, 1])])

    cluster_dist = mydist2(c1_m, c2_m)
    if cluster_dist > 1000:
        return False

    if cluster_dist < 10:
        return True

    tmp_candidate = np.concatenate([candidate, cluster2])

    # Clusters are very far away
    missing_channels = np.setxor1d(np.arange(min(tmp_candidate[:, 0]), max(tmp_candidate[:, 0])),
                                   tmp_candidate[:, 0])

    if len(missing_channels) > len(tmp_candidate):
        return False

    pear_old = pearsonr(candidate[:, 1], candidate[:, 0])
    pear = pearsonr(tmp_candidate[:, 1], tmp_candidate[:, 0])

    if pear[0] <= pear_old[0] and (pear[1] <= pear_old[1]):
        return True

    r1 = __ir(candidate[:, :2])
    r2 = __ir(tmp_candidate[:, :2])

    return r2 < r1


def split(ransac, candidate, candidates, ptracks, g):
    if np.sum(~ransac.inlier_mask_) > 0.2 * len(candidate):

        # print 'Candidate {} will be split'.format(g1)
        split_candidate_1 = candidate[ransac.inlier_mask_]
        split_candidate_2 = candidate[~ransac.inlier_mask_]
        ransac.fit(split_candidate_1[:, 0].reshape(-1, 1), split_candidate_1[:, 1])
        candidate_1, valid_1 = fit(split_candidate_1, ransac.inlier_mask_)

        if valid_1:
            candidates.append(candidate_1)

        ransac.fit(split_candidate_2[:, 0].reshape(-1, 1), split_candidate_2[:, 1])
        candidates_2, valid_2 = fit(split_candidate_2, ransac.inlier_mask_)

        if valid_2:
            candidates.append(candidates_2)

        if not valid_1 and not valid_2:
            ransac.fit(candidate[:, 0].reshape(-1, 1), candidate[:, 1])
            candidate, valid = fit(candidate, ransac.inlier_mask_)

            if valid:
                ptracks.append(candidate)
    else:
        candidate, valid = fit(candidate, ransac.inlier_mask_)

        if valid:
            ptracks.append(candidate)

    return candidates, ptracks


@timeit
def validate_clusters(data, labelled_clusters, unique_labels, e_thold, min_length):
    candidates = []
    for g in unique_labels:
        c = np.vstack(data[labelled_clusters == g, 0])

        m, intercept, r_value, p, e = linregress(c[:, 1], c[:, 0])

        # if len(c) <= min_length:
        #     print 'not significant'

        if p < 0.05 and e < e_thold and len(c) > min_length:
            c = np.append(c, np.expand_dims(np.full(len(c), g), axis=1), axis=1)
            candidates.append(c)

    return candidates


@timeit
def merge_clusters(clusters):
    ransac = linear_model.RANSACRegressor(residual_threshold=5)
    tracks = []
    for j, candidate in enumerate(clusters):
        g = candidate[:, 3][0]
        for i, track in enumerate(tracks):
            # If beam candidate is similar to candidate, merge it.
            if similar(track, candidate, g, track[:, 3][0]):
                c = np.concatenate([track, candidate])

                ransac.fit(c[:, 0].reshape(-1, 1), c[:, 1])
                candidate, valid = fit(c, ransac.inlier_mask_)
                tracks[i] = candidate
                break
        else:
            ransac.fit(candidate[:, 0].reshape(-1, 1), candidate[:, 1])

            candidates, tracks = split(ransac, candidate, clusters, tracks, g)

    return tracks


@timeit
def fill_clusters(tracks, test_img, missing_thold, r_thold, min_span_thold):
    filled_tracks = []

    for i, c in enumerate(tracks):
        group = c[c[:, 3] > 0][:, 3][0]
        missing_channels = np.setxor1d(np.arange(min(c[:, 0]), max(c[:, 0])),
                                       c[:, 0])

        r = len(missing_channels) / float(len(missing_channels) + len(c))
        if r >= missing_thold:
            continue

        m, intercept, r_value, p, e = linregress(c[:, 1], c[:, 0])

        if r_value < r_thold and len(np.unique(c[:, 1])) > min_span_thold:
            c3 = fill(test_img, c)
            filled_tracks.append(c3)
    return filled_tracks


@timeit
def cluster_leaves(leaves, distance_thold=30.):
    cluster_data = np.vstack(
        [(cluster, i, x, y, p) for i, (cluster, best_gs, ratio, x1, x2, y1, y2, n, x, y, p) in enumerate(leaves)])

    cluster_labels, unique_labels = h_cluster_euclidean(cluster_data, distance_thold)

    return cluster_labels, unique_labels, cluster_data


@timeit
def pre_process_data(test_image):
    ndx = np.column_stack(np.where(test_image > 0.))
    ndx = np.append(ndx, np.expand_dims(test_image[test_image > 0.], axis=1), axis=1)
    return ndx


@timeit
def build_tree(ndx, leave_size=30, n_axis=2):
    """
    Build a 2D or 3D kd tree.

    At a high SNR, a 3D kd tree is slow

    :param ndx:
    :param leave_size:
    :param n_axis:
    :return:
    """

    return KDTree(ndx[:, :n_axis], leave_size)


def visualise_ir(data, group, ratio2):
    sample = data[:, 1]
    channel = data[:, 0]

    ms, mc = np.mean(sample), np.mean(channel)
    x = sample - ms
    y = channel - mc

    coords = np.flip(np.swapaxes(data[:, :2] - np.mean(data[:, :2], axis=0), 0, -1), 0)

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]  # Eigenvector with the second largest eigenvalue

    scale = 6

    ax = plt.axes()
    plt.plot([x_v1 * -scale * 1 + ms, x_v1 * scale * 1 + ms],
             [y_v1 * -scale * 1 + mc, y_v1 * scale * 1 + mc], color='black')
    plt.plot([x_v2 * -scale + ms, x_v2 * scale + ms],
             [y_v2 * -scale + mc, y_v2 * scale + mc], color='blue')

    plt.plot(x + ms, y + mc, '.', markersize=12, zorder=1)

    x1, y1, x2, y2 = x_v1 * -scale * 1 + ms, y_v1 * -scale * 1 + mc, x_v1 * scale * 1 + ms, y_v1 * scale * 1 + mc
    x12, y12, x22, y22 = x_v2 * -scale * 1 + ms, y_v2 * -scale * 1 + mc, x_v2 * scale * 1 + ms, y_v2 * scale * 1 + mc
    mem = []
    print
    for point in data:
        x_, y_ = point[1], point[0]
        d1, d2 = nearest2(x_, y_, x1, y1, x2, y2), nearest2(x_, y_, x12, y12, x22, y22)

        t = (d1 + d2)

        a = 1. - (d2 / t)
        if d1 < d2:
            a = 1. - (d1 / t)

        if a <= 0.7:
            # print d1, d2, d1 / d2
            score = 1. - (d2 / t)
            plt.plot(point[1], point[0], 'o', color='r', markersize=10, zorder=-1)
            ax.text(x_ + 0.2, y_ - 0.2, round(score, 2), color='r', fontsize=12, ha='center', va='center')
            # print x_, y_, 'red'
        else:
            if d2 > d1:
                plt.plot(x_, y_, 'o', color='k', markersize=10, zorder=-1)
                # print x_, y_, 'k'
            else:
                plt.plot(x_, y_, 'o', color='b', markersize=10, zorder=-1)
                # print x_, y_, 'b'

            ax.text(x_ + 0.2, y_ + 0.2, round(d2, 2), color='b', fontsize=12, ha='center', va='center')
            ax.text(x_ + 0.2, y_ - 0.2, round(d1, 2), color='k', fontsize=12, ha='center', va='center')
            mem.append(np.array([y_, x_]))

    ratio = __ir(np.vstack(mem))

    ax.text(0.05, 0.95,
            'Group: {}\nRatio: {:0.5f}\nCalculated Ratio: {:0.5f}'.format(group, ratio, ratio2), color='k',
            weight='bold',
            fontsize=15,
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)

    line_1 = (x1, y1), (x2, y2)
    line_2 = (x12, y12), (x22, y22)

    print data[membership(line_1, line_2, data, membership_ratio=0.7), :].shape

    plt.show()
