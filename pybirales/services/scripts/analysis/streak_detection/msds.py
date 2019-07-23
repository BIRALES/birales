import matplotlib.patches as patches
import scipy.spatial.distance as dist
from matplotlib.path import Path
from scipy.spatial import KDTree
from scipy.stats import linregress, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import fclusterdata
from numba import njit
from visualisation import *
from copy import copy
from scipy.spatial import distance
import pandas as pd


# @profile
@njit(fastmath=True)
def mydist(p1, p2):
    diff = p1[:2] - p2[:2]  # dx = p1[1] - p2[1],  dy = p1[0] - p2[0]
    # print p1, p2
    if diff[1]:  # if dx and dy are not 0.
        if np.arctan(diff[0] / diff[1]) > 0:
            return 10000.

    return np.vdot(diff, diff) ** 0.5


@njit(fastmath=True)
def mydist2(p1, p2):
    diff = p1[:3] - p2[:3]  # dx = p1[1] - p2[1],  dy = p1[0] - p2[0]
    # print p1, p2
    if diff[1]:  # if dx and dy are not 0.
        if np.arctan(diff[0] / diff[1]) > 0:
            return 10000.

    return np.vdot(diff, diff) ** 0.5


def mydist3(p1, p2):
    diff = p1[:3] - p2[:3]  # dx = p1[1] - p2[1],  dy = p1[0] - p2[0]
    # print p1, p2
    if diff[1]:  # if dx and dy are not 0.
        deg = np.abs(np.rad2deg(np.arctan(diff[0] / diff[1])))
        if not (0 <= deg <= 60):
            return 10000.

    return np.vdot(diff, diff) ** 0.5


def h_cluster(X, threshold):
    cluster_labels = fclusterdata(X, threshold, metric=mydist2, criterion='distance')

    u, c = np.unique(cluster_labels, return_counts=True)
    unique_groups = u[c > 2]

    return cluster_labels, unique_groups


def h_cluster2(X, threshold):
    cluster_labels = fclusterdata(X[['channel', 'sample', 'snr']], threshold, metric=mydist2, criterion='distance')

    u, c = np.unique(cluster_labels, return_counts=True)
    # unique_groups = u[c > 2]

    return cluster_labels, u


def _partition(data, x1, x2, y1, y2):
    ys = data[:, 0]
    partition_y = data[np.logical_and(ys >= y1, ys <= y2)]
    return partition_y[np.logical_and(partition_y[:, 1] >= x1, partition_y[:, 1] <= x2)]


def traverse(root, ndx, x1, x2, y1, y2):
    rectangles = []
    leaves = []
    if root:
        if not isinstance(root, KDTree.leafnode):
            rectangles.append((root.split_dim, root.split, x1, x2, y1, y2, root.children))

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
                partition = _partition(ndx, x1, x2, y1, y2)

                # rectangles.append((bb, x1, x2, y1, y2, root.children))
                leaves.append((partition, x1, x2, y1, y2, root.children))

    return rectangles, leaves


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


def nearest2(m1, c1, m2, c2, x, y, i=None):
    # np.seterr(all='raise')
    if m1 == np.inf:
        line1_x = 0
        line1_y = y
    elif m1 == 0:
        line1_x = x
        line1_y = 0
    else:
        line1_x = (y - c1) / m1
        line1_y = m1 * x + c1

    if m2 == 0.:
        line2_x = x
        line2_y = 0
    elif m2 == np.inf:
        line2_x = 0
        line2_y = y
    else:
        line2_x = (y - c2) / m2
        line2_y = m2 * x + c2

    d1 = np.hypot(line1_y - y, line1_x - x)
    d2 = np.hypot(line2_y - y, line2_x - x)
    t = d1 + d2

    if t == 0:
        return 1, 1
    # print m1, c1, m2, c2, x, y, d1, d2, t
    if d1 < d2:
        return 1, 1. - (d1 / t)
    return 2, 1. - (d2 / t)


def nearest(m1, c1, m2, c2, x, y, i=None):
    if m1 == np.inf:
        line1_x = 0
        line1_y = y
    else:
        line1_x = (y - c1) / m1
        line1_y = m1 * x + c1

    if m2 == 0.:
        line2_x = x
        line2_y = y
    else:
        line2_x = (y - c2) / m2
        line2_y = m2 * x + c2

    d1 = np.hypot(line1_y - y, line1_x - x)
    d2 = np.hypot(line2_y - y, line2_x - x)
    t = d1 + d2
    # print m1, c1, m2, c2, x, y, d1, d2, t
    if d1 < d2:
        return 1, 1. - (d1 / t)
    return 2, 1. - (d2 / t)


def __inertia_ratio(x, y, i=0):
    if i == 0:
        return 3
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]

    p1 = np.array([x_v1 * -1 * 1, y_v2 * -1 * 1])
    p2 = np.array([x_v1 * 1 * 1, y_v2 * 1 * 1])
    diff = p1 - p2

    pa1 = np.vdot(diff, diff) ** 0.5 + 0.00001
    p1 = np.array([x_v2 * -1 * 1, y_v1 * 1 * 1])
    p2 = np.array([x_v2 * 1 * 1, y_v1 * -1 * 1])
    diff = p1 - p2
    pa2 = np.vdot(diff, diff) ** 0.5 + 0.00001
    scale = 6
    v1_1 = np.array([x_v1 * -scale * 2, y_v1 * -scale * 2])
    v1_2 = np.array([x_v1 * scale * 2, y_v1 * scale * 2])

    v2_1 = np.array([x_v2 * -scale, y_v2 * -scale])
    v2_2 = np.array([x_v2 * scale, y_v2 * scale])

    # if v1_1[0] + v1_2[0] == 0.:
    #     m1, c1 = 10, 0.
    #
    #     if pa1 == 0:
    #         pa1 += 0.0000001
    #
    #     # return 0.169
    # else:
    #     m1, c1 = get_line_eq(v1_1, v1_2)

    m1, c1 = get_line_eq(v1_1, v1_2)

    m2, c2 = get_line_eq(v2_1, v2_2)
    pa1n = 1.
    pa2n = 1.
    for z, (x_, y_) in enumerate(zip(x, y)):

        line, d = nearest2(m1, 0.000001, m2, 0.000001, x_, y_, i)
        if line == 1:
            pa1n += d
        else:
            pa2n += d

    # if i == 4:
    #     scale = 6
    #     fig, ax = plt.subplots(1)
    #     plt.plot([x_v1 * -scale * 2, x_v1 * scale * 2],
    #              [y_v1 * -scale * 2, y_v1 * scale * 2], color='black')
    #     plt.plot([x_v2 * -scale, x_v2 * scale],
    #              [y_v2 * -scale, y_v2 * scale], color='blue')
    #
    #     plt.plot(x, y, '.')
    #     plt.plot(v1_1[0], v1_1[1], '+', color='red', markersize=16)
    #     plt.plot(v1_2[0], v1_2[1], 'x', color='red', markersize=16)
    #
    #     plt.plot(v2_1[0], v2_1[1], '+', color='g', markersize=16)
    #     plt.plot(v2_2[0], v2_2[1], 'x', color='g', markersize=16)
    #
    #     # plt.plot(x_v1, y_v1, 'x', color='g', markersize=16)
    #     # plt.plot(x_v2, y_v2, 'o', color='g', markersize=16)
    #
    #     m1, c1 = get_line_eq(v1_1, v1_2)
    #     m2, c2 = get_line_eq(v2_1, v2_2)
    #     pa1n = 1
    #     pa2n = 1
    #     plt.show()
    #     for x_, y_ in zip(x, y):
    #         line, d = nearest2(m1, 0, m2, 0, x_, y_)
    #         if line == 1:
    #             pa1n += d
    #             plt.plot(x_, y_, 'o', color='k', markersize=10)
    #         else:
    #             pa2n += d
    #             plt.plot(x_, y_, 'o', color='b', markersize=10)
    #         ax.text(x_ + 0.5, y_ + 0.5, round(d, 2), color='k', weight='bold',
    #                 fontsize=8, ha='center', va='center')
    #     # ax.text(v1_1[0]+0.5, v1_1[1]+0.5, i+','+j, color='k', weight='bold',
    #     #         fontsize=12, ha='center', va='center')
    #
    #     plt.show()
    # # print i, evecs[1, sort_indices[0]]/evecs[0, sort_indices[0]], pa2 / pa1, (pa2n / pa1n * pa2/pa1)
    # if i > 1360 and i < 1380:
    #     print i, (pa2n / pa1n * pa2 / pa1), np.abs(evecs[1, sort_indices[0]] / evecs[0, sort_indices[0]])
    return (pa2n / pa1n * pa2 / pa1)
    # return np.abs(evecs[1, sort_indices[0]] / evecs[0, sort_indices[0]])


def __get_shape_factor(x, y):
    """
    Return values related to the shape based on x and y.

    Parameters
    ----------
    x : array_like
        An array of x coordinates.
    y : array_like
        An array of y coordinates.

    Returns
    -------
    area : float
        Area of the border.
    perimeter : float
        Perimeter of the border.
    x_center : float
        X center coordinate.
    y_center : float
        Y center coordinate.
    distances : numpy.ndarray
        Distances from the center to each border element.
    """

    # Area.
    xyxy = (x[:-1] * y[1:] - x[1:] * y[:-1])
    A = 1. / 2. * np.sum(xyxy)

    # X and Y center.
    one_sixth_a = 1. / (6. * A)
    x_center = one_sixth_a * np.sum((x[:-1] + x[1:]) * xyxy)
    y_center = one_sixth_a * np.sum((y[:-1] + y[1:]) * xyxy)

    # Perimeter.
    perimeter = np.sum(np.sqrt((x[1:] - x[:-1]) ** 2 +
                               (y[1:] - y[:-1]) ** 2))

    # Distances from the center.
    distances = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

    return np.abs(A), perimeter, x_center, y_center, distances


def __plot_leave(ax, x1, y1, x2, y2, i, score, positive, cluster=None, noise=None):
    color = 'r'
    zorder = 1
    lw = 1
    if positive:
        color = 'g'
        zorder = 2
        lw = 2
    if np.any(cluster):
        ax.plot(cluster[:, 1], cluster[:, 0], 'g.', zorder=3)
        ax.plot(noise[:, 1], noise[:, 0], 'r.', zorder=3)
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=lw, edgecolor=color, facecolor='none',
                             zorder=zorder)

    # Add the patch to the Axes
    ax.add_patch(rect)

    ax.text(x1 + 0.5 * (x2 - x1), y1 + 0.5 * (y2 - y1), score, color='k', weight='bold',
            fontsize=10, ha='center', va='center')

    ax.text(x1 + 0.95 * (x2 - x1), y1 + 0.95 * (y2 - y1), i, color='k', fontsize=8, ha='right', va='top')


def cos_sim(cluster1, cluster2):
    cluster_m1, cluster_c1, _, _, _ = linregress(cluster1[:, 0], cluster1[:, 1])
    cluster_m2, cluster_c2, _, _, _ = linregress(cluster2[:, 0], cluster2[:, 1])

    return dist.cosine([cluster_m1, cluster_c1], [cluster_m2, cluster_c2]) < 0.1


def cluster_merge(cluster1, cluster2):
    merged = np.vstack([cluster1, cluster2])

    return merged

def cluster_merge_pd(cluster1, cluster2):

    return pd.concat([cluster1, cluster2])



def __plot_candidate(ndx, candidate, i):
    y1, y2 = min(candidate[:, 0]), max(candidate[:, 0])
    x1, x2 = min(candidate[:, 1]), max(candidate[:, 1])

    data = _partition(ndx, x1, x2, y1, y2)
    fig, ax = plt.subplots(1)

    ax.plot(data[:, 1], data[:, 0], '.', zorder=1)
    ax.text(x2, y2, 'Candidate {}'.format(i), color='k', fontsize=40, va="top", ha="right")
    ax.plot(candidate[:, 1], candidate[:, 0], 'o', markersize=10, zorder=0, markerfacecolor='w', lw=2)
    ax.grid(color='gray', linestyle='-', linewidth=1, which='minor', alpha=0.2)
    ax.grid(which='minor', alpha=0.2)

    ax.set_xticks(np.arange(x1, x2), minor=True)
    ax.set_yticks(np.arange(y1, y2), minor=True)

    # ax.set_ylim(y1, y2)
    # ax.set_xlim(x1, x2)

    # plt.show()


def __plot_candidates(candidates):
    for i, candidate in enumerate(candidates):
        fig, ax = plt.subplots()
        __plot_candidate(ax, candidate, i)


def __plot_track(ndx, track, candidates, i):
    cs = []
    for j, c in enumerate(candidates):
        c = np.append(c, np.expand_dims(np.full(shape=len(c), fill_value=j), axis=1), axis=1)
        cs.append(c)

    candidate_data = np.vstack(cs)

    track_x, track_y = track
    y1, y2 = min(track_y), max(track_y)
    x1, x2 = min(track_x), max(track_x)

    ndx_data = _partition(ndx, x1, x2, y1, y2)
    data = _partition(candidate_data, x1, x2, y1, y2)
    fig, ax = plt.subplots(1)

    ax.plot(ndx_data[:, 1], ndx_data[:, 0], '.', zorder=1)
    ax.plot(track_x, track_y, '.', color='k', zorder=1)

    title = 'Track {}'.format(i)

    if np.any(data):
        title += ', Candidate {}'.format(data[0, 4])
    else:
        title += ', No Candidate'

    ax.text(x2, y2, title, color='k', fontsize=40, va="top", ha="right")
    ax.plot(data[:, 1], data[:, 0], 'o', markersize=10, zorder=0, markerfacecolor='w', lw=2)
    ax.grid(color='gray', linestyle='-', linewidth=1, which='minor', alpha=0.2)
    ax.grid(which='minor', alpha=0.2)

    ax.set_xticks(np.arange(x1, x2), minor=True)
    ax.set_yticks(np.arange(y1, y2), minor=True)

    # ax.set_ylim(y1, y2)
    # ax.set_xlim(x1, x2)
    # plt.show()


def __plot_clusters(fig, ax, tracks, clusters, labels, i):
    # ax =  plt.gca()
    track_x, track_y = tracks[i]

    clusters = np.vstack(clusters)

    y1, y2 = min(track_y), max(track_y)
    x1, x2 = min(track_x), max(track_x)
    #
    title = 'Track {}'.format(i)

    ax.plot(track_x, track_y, '.', color='k', zorder=1)
    ax.grid(color='gray', linestyle='-', linewidth=1, which='minor', alpha=0.2)
    ax.grid(which='minor', alpha=0.2)

    data = _partition(clusters, x1, x2, y1, y2)

    # data = clusters
    cluster_data = np.split(data, np.where(np.diff(data[:, 4]))[0] + 1)
    for c in cluster_data:
        ax.plot(c[:, 1], c[:, 0], 'o', markersize=10, zorder=0, markerfacecolor='w', lw=2)

        if np.any(c):
            title += ', Candidate {}'.format(data[0, 4])
        else:
            title += ', No Candidate'

    ax.text(x2 * 1.1, y2 * 1.1, title, color='k', fontsize=12, va="top", ha="right")
    # ax.set_xticks(np.arange(x1, x2), minor=True)
    # ax.set_yticks(np.arange(y1, y2), minor=True)

    # ax.set_ylim(y1, y2)
    # ax.set_xlim(x1, x2)

    return ax


def fill(test_image, cluster, ransac):
    X = np.arange(min(cluster['channel']), max(cluster['channel']))
    missing = np.setxor1d(X, cluster['channel'])

    if len(missing) > len(cluster):
        return cluster

    group = cluster['group'].iloc[0]
    missing_y = np.setxor1d(np.arange(min(cluster['sample']), max(cluster['sample'])), cluster['sample'])
    missing_x = (missing_y - ransac.estimator_.intercept_) / ransac.estimator_.coef_[0]
    missing_x = missing_x.astype(int)

    y = ransac.predict(missing.reshape(-1, 1))

    y = np.concatenate([y, missing_y])
    missing = np.concatenate([missing, missing_x])

    missing_df = pd.DataFrame(columns=['channel', 'sample', 'snr', 'group', 'group_2'])
    missing_df['channel'] = missing
    missing_df['sample'] = y
    missing_df['snr'] = 1
    missing_df['group'] = -1
    missing_df['group_2'] = group

    # p = test_image[m[:, 0], m[:, 1]]

    return pd.concat([cluster, missing_df])


def fill2(test_image, cluster, ransac):
    X = np.arange(min(cluster[:, 1]), max(cluster[:, 1]))
    missing = np.setxor1d(X, cluster[:, 1])

    if len(missing) > len(cluster):
        return cluster

    group = cluster[:, 4][0]
    missing_y = np.setxor1d(np.arange(min(cluster[:, 0]), max(cluster[:, 0])), cluster[:, 0])
    missing_x = (missing_y - ransac.estimator_.intercept_) / ransac.estimator_.coef_[0]
    missing_x = missing_x.astype(int)

    y = ransac.predict(missing.reshape(-1, 1))

    y = np.concatenate([y, missing_y])
    missing = np.concatenate([missing, missing_x])

    m = np.column_stack((y, missing)).astype(int)

    # p = test_image[m[:, 0], m[:, 1]]
    p = np.ones(shape=len(m))
    m = np.append(m, np.expand_dims(p, axis=1), axis=1)

    # cluster group is -1 since these did not exist
    m = np.append(m, np.expand_dims(np.full(shape=len(m), fill_value=-1), axis=1), axis=1)

    m = np.append(m, np.expand_dims(np.full(shape=len(m), fill_value=group), axis=1), axis=1)

    return np.concatenate([cluster, m], axis=0)

def fit(cluster, inliers):
    d = cluster[inliers]

    if len(np.unique(d['sample'])) == 1 or len(np.unique(d['channel'])) == 1:
        return cluster, False

    pear = pearsonr(d['sample'], d['channel'])

    if pear[1] < 0.05 and pear[0] < -0.9 and len(d) >= 10:
        return d, True

    return cluster, False


def similar(candidate, cluster2, i=None, j=None):
    cluster_m1, cluster_c1, _, _, _ = linregress(candidate['sample'], candidate['channel'])
    cluster_m2, cluster_c2, _, _, _ = linregress(cluster2['sample'], cluster2['channel'])

    # if dist.cosine([cluster_m1, cluster_c1], [cluster_m2, cluster_c2]) < 1e-8:
    #     tmp_candidate = cluster_merge(candidate, cluster2)
    #
    #     pear_old = pearsonr(candidate['sample'], candidate['channel'])
    #     pear = pearsonr(tmp_candidate['sample'], tmp_candidate['channel'])
    #
    #     if pear[0] <= pear_old[0] or (pear[1] <= pear_old[1] and np.abs((pear[0] - pear_old[0]) / pear[0]) < 0.01):
    #         return True

    if dist.cosine([cluster_m1, cluster_c1], [cluster_m2, cluster_c2]) < 1e-8:
        tmp_candidate = cluster_merge_pd(candidate, cluster2)

        ratio_old = __inertia_ratio(candidate['channel'], candidate['sample'])
        ratio = __inertia_ratio(tmp_candidate['channel'], tmp_candidate['sample'])

        # if ratio <= ratio_old:
        #     return True

        pear_old = pearsonr(candidate['sample'], candidate['channel'])
        pear = pearsonr(tmp_candidate['sample'], tmp_candidate['channel'])

        if pear[0] <= pear_old[0] or (pear[1] <= pear_old[1]):
            return True

    return False

    # return dist.cosine([cluster_m1, cluster_c1], [cluster_m2, cluster_c2]) < 0.1
