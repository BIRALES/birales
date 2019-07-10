import matplotlib.patches as patches
import scipy.spatial.distance as dist
from matplotlib.path import Path
from scipy.spatial import KDTree
from scipy.stats import linregress, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import fclusterdata
from numba import njit
from visualisation import *

# np.seterr(all='raise')
# @profile
@njit(fastmath = True)
def mydist(p1, p2):
    diff = p1 - p2  # dx = p1[1] - p2[1],  dy = p1[0] - p2[0]

    if diff[1]:  # if dx and dy are not 0.
        if np.arctan(diff[0] / diff[1]) > 0:
            return 1000.

    return np.vdot(diff, diff) ** 0.5


def traverse(root, x1, x2, y1, y2):
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

            left, ll = traverse(root.less, _x1, _x2, _y1, _y2)

            if root.split_dim == 0:
                _y1, _y2 = root.split, y2
            if root.split_dim == 1:
                _x1, _x2 = root.split, x2

            right, lr = traverse(root.greater, _x1, _x2, _y1, _y2)
            rectangles = rectangles + left
            rectangles = rectangles + right

            leaves = leaves + ll
            leaves = leaves + lr

        else:
            d = root.children / ((x2 - x1) * (y2 - y1))
            rectangles.append((d > 0.069, None, x1, x2, y1, y2, root.children))
            leaves.append((d > 0.069, None, x1, x2, y1, y2, root.children))

            # cc = np.column_stack((np.arange(0, 10), np.arange(0, 10)))
            # fclusterdata(cc, 100.0, criterion='distance')

            # print 'Density', d, d > 0.069, 'Area:', ((x2 - x1) * (y2 - y1)), root.children
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
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    return m, c


def nearest(m1, c1, m2, c2, x, y):
    line1_y = m1 * x + c1
    line2_y = m2 * x + c2

    line1_x = (y - c1) / m1
    line2_x = (y - c2) / m2

    d1 = np.hypot(line1_y - y, line1_x - x)
    d2 = np.hypot(line2_y - y, line2_x - x)
    t = d1 + d2
    # print m1, c1, m2, c2, x, y, d1, d2, t
    if d1 < d2:
        return 1, 1. - (d1 / t)
    return 2, 1. - (d2 / t)


def __inertia_ratio(x, y, i=0):
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

    pa1 = np.vdot(diff, diff) ** 0.5
    p1 = np.array([x_v2 * -1 * 1, y_v1 * 1 * 1])
    p2 = np.array([x_v2 * 1 * 1, y_v1 * -1 * 1])
    diff = p1 - p2
    pa2 = np.vdot(diff, diff) ** 0.5
    scale = 6
    v1_1 = np.array([x_v1 * -scale * 2, y_v1 * -scale * 2])
    v1_2 = np.array([x_v1 * scale * 2, y_v1 * scale * 2])

    v2_1 = np.array([x_v2 * -scale, y_v2 * -scale])
    v2_2 = np.array([x_v2 * scale, y_v2 * scale])

    m1, c1 = get_line_eq(v1_1, v1_2)
    m2, c2 = get_line_eq(v2_1, v2_2)
    pa1n = 1.
    pa2n = 1.
    for x_, y_ in zip(x, y):
        line, d = nearest(m1, 0.000001, m2, 0.000001, x_, y_)
        if line == 1:
            pa1n += d
        else:
            pa2n += d

    if i == 7700:
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

        # plt.plot(x_v1, y_v1, 'x', color='g', markersize=16)
        # plt.plot(x_v2, y_v2, 'o', color='g', markersize=16)

        m1, c1 = get_line_eq(v1_1, v1_2)
        m2, c2 = get_line_eq(v2_1, v2_2)
        pa1n = 1
        pa2n = 1
        for x_, y_ in zip(x, y):
            line, d = nearest(m1, 0, m2, 0, x_, y_)
            if line == 1:
                pa1n += d
                plt.plot(x_, y_, 'o', color='k', markersize=10)
            else:
                pa2n += d
                plt.plot(x_, y_, 'o', color='b', markersize=10)
            ax.text(x_ + 0.5, y_ + 0.5, round(d, 2), color='k', weight='bold',
                    fontsize=8, ha='center', va='center')

        plt.show()
    # print i, evecs[1, sort_indices[0]]/evecs[0, sort_indices[0]], pa2 / pa1, (pa2n / pa1n * pa2/pa1)

    return (pa2n / pa1n * pa2 / pa1)
    return np.abs(evecs[1, sort_indices[0]] / evecs[0, sort_indices[0]])


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


def similar(candidate, cluster2, i):
    cluster2_group = cluster2[:, 3][0].astype(int)
    candidate_group = candidate[:, 3][0].astype(int)
    cluster_m1, cluster_c1, _, _, _ = linregress(candidate[:, 0], candidate[:, 1])
    cluster_m2, cluster_c2, _, _, _ = linregress(cluster2[:, 0], cluster2[:, 1])

    # return dist.cosine(candidate, cluster2)
    # print dist.cosine([cluster_m1, cluster_c1], [cluster_m2, cluster_c2]), i, cosine_similarity(candidate, cluster2)
    # return dist.cosine([cluster_m1, cluster_c1], [cluster_m2, cluster_c2]) < 0.05

    # return cosine_similarity(candidate[:, 1].reshape(-1, 1), cluster2[:, 1].reshape(-1, 1))
    if dist.cosine([cluster_m1, cluster_c1], [cluster_m2, cluster_c2]) < 0.05:
        tmp_candidate = cluster_merge(candidate, cluster2)

        pear_old = pearsonr(candidate[:, 0], candidate[:, 1])
        pear = pearsonr(tmp_candidate[:, 0], tmp_candidate[:, 1])

        if pear[0] < pear_old[0] and pear[1] < 0.05:
            return True
    return False

    # return dist.cosine([cluster_m1, cluster_c1], [cluster_m2, cluster_c2]) < 0.1


def cluster_merge(cluster1, cluster2):
    cluster2_group = cluster1[:, 3][0].astype(int)
    candidate_group = cluster2[:, 3][0].astype(int)
    # print cluster2_group, candidate_group, cluster1.shape, cluster2.shape
    # if cluster2_group == 0 and candidate_group == 174:
    #     print cluster2_group, candidate_group, cluster1.shape, cluster2.shape
    merged = np.vstack([cluster1, cluster2])
    # print cluster2_group, candidate_group, cluster1.shape, cluster2.shape, '==>', merged.shape

    return merged

    # check that correlation improves...
    # pear1 = pearsonr(cluster1[:, 0], cluster1[:, 1])
    # pear2 = pearsonr(cluster1[:, 0], cluster1[:, 1])

    # if


def __plot_candidates(ax, candidates):
    for i, candidate in enumerate(candidates):
        codes = [Path.LINETO for c in candidate]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        candidate = candidate[np.lexsort((candidate[:, 1], candidate[:, 0]))]

        verts = np.fliplr(candidate[:, 0:2])
        # ax.add_patch(patches.PathPatch(Path(verts, codes), facecolor='orange', lw=2))

        x = np.mean(verts[:, 1])
        y = np.mean(verts[:, 0])
        ax.text(x, y, i, color='k', fontsize=22, ha='right', va='top')

        ax.plot(candidate[:, 1], candidate[:, 0], 'o', markersize=10, zorder=2, markerfacecolor='w', lw=2)
