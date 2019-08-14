import numpy as np
from numba import njit
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import KDTree
from scipy.stats import linregress, pearsonr
from skimage.filters import threshold_otsu
from sklearn import linear_model

from util import timeit


# @profile
@njit(fastmath=True)
def mydist2(p1, p2):
    diff = p1[:3] - p2[:3]  # dx = p1[1] - p2[1],  dy = p1[0] - p2[0]
    # print p1, p2
    if diff[1]:  # if dx and dy are not 0.
        if np.arctan(diff[0] / diff[1]) > 0:
            return 10000.

    return np.vdot(diff, diff) ** 0.5


def h_cluster(X, dis_threshold, min_length=2):
    cluster_labels = fclusterdata(X, dis_threshold, metric=mydist2, criterion='distance')

    u, c = np.unique(cluster_labels, return_counts=True)
    unique_groups = u[c > min_length]

    return cluster_labels, unique_groups


def h_cluster_euclidean(X, dis_threshold):
    cluster_labels = fclusterdata(X[:, 2:4], dis_threshold, metric='euclidean', criterion='distance')

    u, c = np.unique(cluster_labels, return_counts=True)

    return cluster_labels, u


def _partition(data, x1, x2, y1, y2):
    ys = data[:, 0]
    partition_y = data[np.logical_and(ys >= y1, ys <= y2)]
    return partition_y[np.logical_and(partition_y[:, 1] >= x1, partition_y[:, 1] <= x2)]


# @profile
# @timeit
def traverse(root, ndx, x1, x2, y1, y2):
    rectangles = []
    leaves = []
    if root:
        if not isinstance(root, KDTree.leafnode):
            # rectangles.append((root.split_dim, root.split, x1, x2, y1, y2, root.children))

            _x1, _x2, _y1, _y2 = x1, x2, y1, y2
            if root.split_dim == 0:
                _y1, _y2 = y1, root.split
            if root.split_dim == 1:
                _x1, _x2 = x1, root.split

            partition = _partition(ndx, _x1, _x2, _y1, _y2)
            left, ll = traverse(root.less, partition, _x1, _x2, _y1, _y2)

            if root.split_dim == 0:
                _y1, _y2 = root.split, y2
            if root.split_dim == 1:
                _x1, _x2 = root.split, x2

            partition = _partition(ndx, _x1, _x2, _y1, _y2)

            right, lr = traverse(root.greater, partition, _x1, _x2, _y1, _y2)
            rectangles = rectangles + left
            rectangles = rectangles + right

            leaves = leaves + ll
            leaves = leaves + lr

        else:
            if root.children > 3:
                labels, _ = h_cluster(ndx, dis_threshold=3.0, min_length=2)
                ratio, best_gs, cluster, rejected_data = __best_group(labels, ndx, bbox=(x1, x2, y1, y2))
                n = len(cluster)
                mesg = 'B:{}\n N:{}\nI:{:0.3f}'.format(best_gs, n, ratio)

                if best_gs > 0:
                    if n >= 10:
                        leaves.append(
                            (cluster, best_gs, mesg, x1, x2, y1, y2, n, np.mean([x1, x2]), np.mean([y1, y2]),
                             np.mean(cluster[:, 2])))
                    else:
                        # if len(np.unique(cluster[:, 0])) > 1 and len(np.unique(cluster[:, 1])) > 1:
                        pear = pearsonr(cluster[:, 1], cluster[:, 0])

                        if pear[1] <= 0.05:
                            leaves.append(
                                (cluster, best_gs, mesg, x1, x2, y1, y2, n, np.mean([x1, x2]), np.mean([y1, y2]),
                                 np.mean(cluster[:, 2])))
                            # print len(leaves), n, pear
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
    bbox_area = (x2 - x1) * (y2 - y1)
    for j, g in enumerate(u_groups):
        c = bb[np.where(fclust1 == g)]
        ratio, c2 = __inertia_ratio(c, i, g)
        cluster_area = (max(c2[:, 0]) - min(c2[:, 0])) * (max(c2[:, 1]) - min(c2[:, 1]))
        thres = len(bb) / bbox_area * cluster_area
        if ratio < best_ratio:
            best_ratio = ratio
            best_group = g
            best_data = c2
        # elif len(c2) > 0.5
        else:
            rejected_data.append((c2, ratio))

    if not best_group:
        return best_ratio, -1, bb, rejected_data
    # todo - merge best groups

    return best_ratio, best_group, best_data, rejected_data


def PreorderTraversal(root):
    splits = []
    if root:
        # splits.append(root.children)
        if not isinstance(root, KDTree.leafnode):
            splits.append((root.split_dim, root.split))
            splits = splits + PreorderTraversal(root.less)
            splits = splits + PreorderTraversal(root.greater)
        else:
            pass
    return splits


# @profile
# @njit
def nearest4(m1, m2, x, y, i=None):
    d1, d2 = nearest(m1, m2, x, y)
    t = d1 + d2 + 0.000000000001

    mask = d1 < d2

    a = 1. - (d2 / t)
    a[mask] = (1. - (d1 / t))[mask]

    return a > 0.7


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
def __inertia_ratio(data, ii=0, g=0, visualise=None):
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

    s = nearest4(m1, m2, coords[0, :], coords[1, :])

    subset = data[s, :]

    if len(subset) < 3:
        return 30., data

    ratio2 = __ir(subset[:, :2])

    return ratio2, subset


# @njit
def __ir(data, i=0, g=0):
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


def fill(test_image, cluster, ransac, u_test_image, fill_thickness=False):
    channel_width = estimate_thickness(cluster, axis=1)
    sample_width = estimate_thickness(cluster, axis=0)

    add_missing = channel_width > 1 and sample_width > 1

    # if not add_missing:
    #     return cluster

    sample = cluster[:, 1]
    channel = cluster[:, 0]
    missing_channels = np.setxor1d(np.arange(min(channel), max(channel)), channel)
    n = len(missing_channels)
    if n > len(cluster):
        return cluster

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

    # filter false positives

    n = np.mean(u_test_image)
    s = np.std(u_test_image)
    t = n + 5 * s

    t = np.mean(test_image[test_image > 0])

    print t, np.mean(test_image[test_image > 0]) * 0.5
    # combined = combined[combined[:, 3] < 0][combined[:, 2] > n]

    return combined[np.logical_or(combined[:, 3] > -2, combined[:, 2] > t)]


def fit(cluster, in_liers):
    d = cluster[in_liers]

    if len(np.unique(d[:, 1])) == 1 or len(np.unique(d[:, 0])) == 1:
        return cluster, False

    pear = pearsonr(d[:, 1], d[:, 0])

    if pear[1] < 0.05 and pear[0] < -0.9 and len(d) >= 10:
        return d, True

    return cluster, False


def similar(candidate, cluster2, i=None, j=None):
    c1_m = np.array([np.mean(candidate[:, 0]), np.mean(candidate[:, 1])])
    c2_m = np.array([np.mean(cluster2[:, 0]), np.mean(cluster2[:, 1])])

    # print i, j, '=>', mydist2(c1_m, c2_m)
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

    if r2 < r1:
        return True

    return False


def split(ransac, candidate, candidates, ptracks, g):
    if np.sum(~ransac.inlier_mask_) > 0.2 * len(candidate):

        # print 'Candidate {} will be split'.format(g1)
        split_candidate_1 = candidate[ransac.inlier_mask_]
        split_candidate_2 = candidate[~ransac.inlier_mask_]
        ransac.fit(split_candidate_1[:, 0].reshape(-1, 1), split_candidate_1[:, 1])
        candidate_1, valid_1 = fit(split_candidate_1, ransac.inlier_mask_)

        if valid_1:
            # print 'Sub-candidate 1/{} added to candidates'.format(g1)
            candidates.append(candidate_1)

        ransac.fit(split_candidate_2[:, 0].reshape(-1, 1), split_candidate_2[:, 1])
        candidates_2, valid_2 = fit(split_candidate_2, ransac.inlier_mask_)

        if valid_2:
            # print 'Sub-candidate 2/{} added to candidates'.format(g1)

            candidates.append(candidates_2)

        if not valid_1 and not valid_2:
            ransac.fit(candidate[:, 0].reshape(-1, 1), candidate[:, 1])
            candidate, valid = fit(candidate, ransac.inlier_mask_)

            if valid:
                # candidate = fill(test_img, candidate, ransac)

                ptracks.append(candidate)
    else:
        candidate, valid = fit(candidate, ransac.inlier_mask_)

        if valid:
            # candidate = fill(test_img, candidate, ransac)

            ptracks.append(candidate)
            # print 'Candidate {} added to tracks'.format(g1)
        # else:
        #     print 'Candidate {} dropped'.format(g1)

    return candidates, ptracks


@timeit
def validate_clusters(data, labelled_clusters, unique_labels):
    candidates = []
    for g in unique_labels:
        c = np.vstack(data[labelled_clusters == g, 0])

        m, intercept, r_value, p, e = linregress(c[:, 1], c[:, 0])

        if p < 0.05 and e < 0.1 and len(c) > 4:
            c = np.append(c, np.expand_dims(np.full(len(c), g), axis=1), axis=1)
            candidates.append(c)

    return candidates
    # print '{} Groups reduced to {} candidates after filtering 1. Time taken: {:0.3f} seconds'.format(len(u_groups),
    #                                                                                                  len(candidates),
    #                                                                                                  _td(t1))


@timeit
def merge_clusters(clusters):
    ransac = linear_model.RANSACRegressor(residual_threshold=5)
    tracks = []
    for j, candidate in enumerate(clusters):
        g = candidate[:, 3][0]
        for i, track in enumerate(tracks):
            # If beam candidate is similar to candidate, merge it.
            # print 'Comparing candidate {} with track {}'.format(g, i)
            if similar(track, candidate, g, track[:, 3][0]):
                c = np.concatenate([track, candidate])

                ransac.fit(c[:, 0].reshape(-1, 1), c[:, 1])
                candidate, valid = fit(c, ransac.inlier_mask_)
                tracks[i] = candidate
                break
        else:
            # print 'Candidate {} is unique'.format(g)
            ransac.fit(candidate[:, 0].reshape(-1, 1), candidate[:, 1])

            candidates, tracks = split(ransac, candidate, clusters, tracks, g)

    return tracks


@timeit
def fill_clusters(tracks, test_img, u_test_image):
    filled_tracks = []
    ransac = linear_model.RANSACRegressor(residual_threshold=5)
    for i, c in enumerate(tracks):
        m, intercept, r_value, p, e = linregress(c[:, 1], c[:, 0])
        ratio = __ir(c[:, :2])

        time_span = len(np.unique(c[:, 1]))
        # print i, c[:, 3][0], 'R:{:0.5f} P:{:0.5f} E:{:0.5f} I:{:0.5f} T:{}'.format(r_value, p, e, ratio, time_span)

        missing_channels = np.setxor1d(np.arange(min(c[:, 0]), max(c[:, 0])),
                                       c[:, 0])

        r = len(missing_channels) / float(len(missing_channels) + len(c))
        if r >= 0.45:
            continue

        if r_value < -0.95 and time_span > 15:
            ransac = ransac.fit(c[:, 0].reshape(-1, 1), c[:, 1])
            c3 = fill(test_img, c, ransac, u_test_image)
            filled_tracks.append(c3)
            # print 'Track ', i, 'power', np.mean(c[:, 2])

    return filled_tracks


@timeit
def cluster_leaves(leaves):
    cluster_data = np.vstack(
        [(cluster, i, x, y, p) for i, (cluster, best_gs, ratio, x1, x2, y1, y2, n, x, y, p) in enumerate(leaves)])

    cluster_labels, unique_labels = h_cluster_euclidean(X=cluster_data, dis_threshold=30.)

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
