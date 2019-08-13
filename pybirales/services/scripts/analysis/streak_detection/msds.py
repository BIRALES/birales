import matplotlib.patches as patches
import pandas as pd
import scipy.spatial.distance as dist
from numba import njit
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import KDTree
from scipy.stats import linregress, pearsonr

from visualisation import *


# profile = line_profiler.LineProfiler()

@njit(fastmath=True)
def mydist2(p1, p2):
    diff = p1[:3] - p2[:3]  # dx = p1[1] - p2[1],  dy = p1[0] - p2[0]
    # print p1, p2
    if diff[1]:  # if dx and dy are not 0.
        if np.arctan(diff[0] / diff[1]) > 0:
            return 10000.

    return np.vdot(diff, diff) ** 0.5


def h_cluster(X, threshold):
    cluster_labels = fclusterdata(X, threshold, metric=mydist2, criterion='distance')

    u, c = np.unique(cluster_labels, return_counts=True)
    unique_groups = u[c > 2]

    return cluster_labels, unique_groups

# @profile
def h_cluster2(X, threshold):
    cluster_labels = fclusterdata(X, threshold, metric=mydist2, criterion='distance')

    u, c = np.unique(cluster_labels, return_counts=True)
    # unique_groups = u[c > 1]

    return cluster_labels, u


# @profile
def h_cluster3(X, threshold):
    cluster_labels = fclusterdata(X[:, 2:4], threshold, metric=mydist2, criterion='distance')

    u, c = np.unique(cluster_labels, return_counts=True)
    unique_groups = u[c > 1]

    return cluster_labels, unique_groups


def _partition(data, x1, x2, y1, y2):
    ys = data[:, 0]
    partition_y = data[np.logical_and(ys >= y1, ys <= y2)]
    return partition_y[np.logical_and(partition_y[:, 1] >= x1, partition_y[:, 1] <= x2)]


# @profile
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

                partition = _partition(ndx, x1, x2, y1, y2)
                labels = fclusterdata(partition, 3.0, metric=mydist2, criterion='distance')
                ratio, best_gs, cluster, rejected_data = __best_group(labels, partition, bbox=(x1, x2, y1, y2))
                n = len(cluster)
                mesg = 'N:{}\nI:{:0.3f}'.format(n, ratio)
                if best_gs > 0 and n > 3:
                    if len(np.unique(cluster[:, 0])) > 1 and len(np.unique(cluster[:, 1])) > 1:
                        pear = pearsonr(cluster[:, 1], cluster[:, 0])

                        if pear[1] <= 0.05 or n >= 10:
                            leaves.append((cluster, best_gs, mesg, x1, x2, y1, y2, n, np.mean([x1, x2]), np.mean([y1, y2])))
                            # print len(leaves), n, pear
                        else:
                            mesg += '\nS:{:0.3f}'.format(pear[1])
                            rectangles.append((cluster, rejected_data, best_gs, mesg, x1, x2, y1, y2, n))
                    else:
                        mesg += '\nU:{}'.format(len(np.unique(cluster[:, 0])))
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
    # if visualise:
    #     visualise_ir(data[:, 1], data[:, 0], data[:, 2], visualise)
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


def visualise_ir(sample, channel, snr, group):
    x = sample - np.mean(sample)
    y = channel - np.mean(channel)
    n = len(sample)
    df = pd.DataFrame(columns=['channel', 'sample', 'vector', 'distance', 'ratio', 'ratio_n', 'group', 'snr'])
    df['sample'] = sample
    df['channel'] = channel
    df['group'] = group
    df['x'] = x
    df['y'] = y
    df['snr'] = snr

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

    m1, c1 = get_line_eq(v1_1, v1_2)

    m2, c2 = get_line_eq(v2_1, v2_2)

    lines = []
    ds = []
    for i, (x_, y_) in enumerate(zip(x, y)):
        line, d = nearest2(m1, 0, m2, 0, x_, y_)
        lines.append(line)
        ds.append(d)

    df['vector'] = np.array(lines)
    df['distance'] = np.array(ds)
    s = df[df['distance'] > .7]

    ratio2 = __ir(s['sample'], s['channel'])

    scale = 6
    fig, ax = plt.subplots(1)
    plt.plot([x_v1 * -scale * 2, x_v1 * scale * 2],
             [y_v1 * -scale * 2, y_v1 * scale * 2], color='black')
    plt.plot([x_v2 * -scale, x_v2 * scale],
             [y_v2 * -scale, y_v2 * scale], color='blue')

    plt.plot(x, y, '.')
    plt.plot(v1_1[0], v1_1[1], '+', color='red', markersize=16)
    plt.plot(v1_2[0], v1_2[1], 'x', color='red', markersize=16)

    plt.plot(v2_1[0], v2_1[1], '+', color='g', markersize=16)
    plt.plot(v2_2[0], v2_2[1], 'x', color='g', markersize=16)

    m1, c1 = get_line_eq(v1_1, v1_2)
    m2, c2 = get_line_eq(v2_1, v2_2)

    pa1n = 1
    pa2n = 1
    for x_, y_ in zip(x, y):
        line, d = nearest2(m1, 0, m2, 0, x_, y_)
        if d <= .7:
            plt.plot(x_, y_, 'o', color='r', markersize=10)
        else:
            if line == 1:
                pa1n += d
                plt.plot(x_, y_, 'o', color='k', markersize=10)
            else:
                pa2n += d
                plt.plot(x_, y_, 'o', color='b', markersize=10)
        ax.text(x_ + 0.5, y_ + 0.5, round(d, 2), color='k', weight='bold',
                fontsize=12, ha='center', va='center')

    ratio1 = (pa2n / pa1n * pa2 / pa1)
    ax.text(np.median(x), np.max(y) * 1.5,
            'Group: {}\nRatio: {:0.5f}\nRatio2: {:0.5f}'.format(group, ratio1, ratio2),
            color='k',
            weight='bold', fontsize=15, ha='center', va='center')

    print group, m1, m2, pa1, pa2, ratio2, len(s), n, ratio1

    plt.show()


def __plot_leave(ax, x1, y1, x2, y2, i, score, positive, positives=None, negatives=None):
    color = 'r'
    zorder = 1
    lw = 1
    if positive:
        color = 'g'
        zorder = 2
        lw = 2
    if np.any(positives):
        ax.plot(positives[:, 1], positives[:, 0], 'g.', zorder=3)

    if negatives > 0:
        if not positive:
            ax.plot(positives[:, 1], positives[:, 0], 'r.', zorder=4)

        # scores = '{}\n'.format(i)
        scores = ''
        colors = ['b', 'm', 'c', 'y', 'g', 'orange', 'indianred']
        for j, (n, ratio) in enumerate(negatives):
            ax.plot(n[:, 1], n[:, 0], '.', zorder=5, color=colors[j])
            scores += '{}: {:0.3f}\n'.format(j, ratio)
        ax.text(x1 * 1.01, y1 + 0.95 * (y2 - y1), scores, color='k', fontsize=10, va='top')

    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=lw, edgecolor=color, facecolor='none',
                             zorder=zorder)

    # Add the patch to the Axes
    ax.add_patch(rect)

    if not negatives:
        ax.text(x1 + 0.5 * (x2 - x1), y1 + 0.5 * (y2 - y1), score, color='k', weight='bold',
                fontsize=10, ha='center', va='center')

    ax.text(x1 + 0.95 * (x2 - x1), y1 + 0.95 * (y2 - y1), i, color='k', fontsize=8, ha='right', va='top', zorder=10)


def cluster_merge(cluster1, cluster2):
    merged = np.vstack([cluster1, cluster2])

    return merged


def cluster_merge_pd(cluster1, cluster2):
    return np.concatenate([cluster1, cluster2])


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
        # filled_1 = np.zeros(shape=(2, na * len(offsets_a)))
        for i, o in enumerate(offsets_a):
            # filled[i * na:i * na + na, :] = predict_y(x, slope, c, o)

            filled[i * na:i * na + na, :] = np.column_stack([channel, sample + o]).astype(int)

    if b > 0:
        # filled_2 = np.zeros(shape=(2, nb * len(offsets_b)))
        for i, o in enumerate(offsets_b):
            # filled[nna + (i * nb):nna + (i * nb + nb), :] = predict_x(y, slope, c, o)

            filled[nna + (i * nb):nna + (i * nb + nb), :] = np.column_stack([channel + o, sample]).astype(int)

    return filled.astype(int)


def fill(test_image, cluster, ransac, fill_thickness=False):
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

    return combined


def fit(cluster, inliers):
    d = cluster[inliers]

    if len(np.unique(d[:, 1])) == 1 or len(np.unique(d[:, 0])) == 1:
        return cluster, False

    pear = pearsonr(d[:, 1], d[:, 0])

    if pear[1] < 0.05 and pear[0] < -0.9 and len(d) >= 10:
        return d, True

    return cluster, False


def similar(candidate, cluster2, i=None, j=None):
    cluster_m1, cluster_c1, _, _, _ = linregress(candidate[:, 1], candidate[:, 0])
    cluster_m2, cluster_c2, _, _, _ = linregress(cluster2[:, 1], cluster2[:, 0])

    if dist.cosine([cluster_m1, cluster_c1], [cluster_m2, cluster_c2]) < 1e-8:
        tmp_candidate = cluster_merge_pd(candidate, cluster2)

        # Clusters are very far away
        missing_channels = np.setxor1d(np.arange(min(tmp_candidate[:, 0]), max(tmp_candidate[:, 0])), tmp_candidate[:, 0])
        n = len(missing_channels)
        if n > len(tmp_candidate):
            return False

        pear_old = pearsonr(candidate[:, 1], candidate[:, 0])
        pear = pearsonr(tmp_candidate[:, 1], tmp_candidate[:, 0])

        if pear[0] <= pear_old[0]: # or (pear[1] <= pear_old[1]):
            return True

    return False
