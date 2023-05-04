"""
This script lists the feature algorithm that will be used to extract
space debris tracks from the filtered data

"""

from astride.utils.edge import EDGE
from astropy.stats import sigma_clipped_stats
from line_profiler import LineProfiler
# from hdbscan import HDBSCAN
from skimage import measure
from skimage.transform import probabilistic_hough_line
from sklearn.cluster import DBSCAN

import pybirales.pipeline.modules.detection.msds.msdsv2 as msds2
from pybirales.pipeline.modules.detection.msds.msds import *
from pybirales.pipeline.modules.detection.msds.util import get_clusters, _validate_clusters, timeit
from pybirales.pipeline.modules.detection.msds.visualisation import *
from receiver import *


def _viz_cluster(bb, fclust1):
    for b, g in zip(bb, fclust1):
        print(b[0], b[1], b[2], g)


def hough_transform(test_image, true_tracks, noise_est, debug, profiler):
    """
    Hough feature detection

    Clustering within the dataset are identified using the Hough transform

    :return:
    """

    # Classic straight-line Hough transform
    # h, thetas, d = hough_line(test_image)

    # edges = canny(test_image, 2, 1, 25)
    test_image[test_image <= 0] = 0
    lines = probabilistic_hough_line(test_image, threshold=10, line_length=5,
                                     line_gap=3)

    clusters = []
    for i, line in enumerate(lines):
        p0, p1 = line
        cluster = generate_line(x1=p0[0], x2=p1[0], y1=p0[1], y2=p1[1], limits=test_image.shape)
        cluster = np.column_stack([cluster, np.full(len(cluster), 1), np.full(len(cluster), i)])
        clusters.append(cluster)

    clusters = _validate_clusters(clusters)

    return clusters


def astride(test_image, true_tracks, noise_est, debug, profiler):
    mean, med, std = sigma_clipped_stats(test_image)
    test_image -= med
    contours = measure.find_contours(test_image, std * 3, fully_connected='high')

    # Quantify shapes of the contours and save them as 'edges'.
    edge = EDGE(contours, min_points=10,
                shape_cut=0.2, area_cut=10,
                radius_dev_cut=0.5,
                connectivity_angle=3.)
    edge.quantify()

    # Filter the edges, so only streak remains.
    edge.filter_edges()

    edge.connect_edges()

    streaks = edge.get_edges()

    clusters = []
    for i, streak in enumerate(streaks):
        cluster = np.column_stack(
            [streak['y'], streak['x'], test_image[streak['y'].astype(int), streak['x'].astype(int)],
             np.full(len(streak['x']), i)])
        clusters.append(cluster)
    #
    # for i, streak in enumerate(streaks):
    #     # cluster = np.column_stack([streak['y'], streak['x'], np.full(len(streak['x']), i)])
    #
    #     cluster_df = pd.DataFrame(columns=['channel', 'sample', 'snr'])
    #     cluster_df['channel'] = streak['y']
    #     cluster_df['sample'] = streak['x']
    #     cluster_df['snr'] = test_image[streak['y'].astype(int), streak['x'].astype(int)]
    #
    #     clusters.append(cluster_df)

    # clusters = _validate_clusters(clusters)

    return clusters


# def hdbscan(test_image, debug, profiler):
#     hdbscan = HDBSCAN(algorithm='prims_kdtree', alpha=1.0, approx_min_span_tree=True,
#                       gen_min_span_tree=False, leaf_size=50, metric='euclidean',
#                       min_cluster_size=5, min_samples=None, p=None)
#     ndx = np.column_stack(np.where(test_image > 0.))
#
#     ndx = np.append(ndx, np.expand_dims(test_image[test_image > 0.], axis=1), axis=1)
#
#     # ndx = pairwise_distances(ndx, metric=sim)
#
#     c_labels = hdbscan.fit_predict(ndx)
#
#     clusters = get_clusters(ndx, c_labels)
#
#     clusters = _validate_clusters(clusters)
#
#     return clusters


def naive_dbscan(test_image, true_tracks, noise_estimate, debug, profiler):
    db_scan = DBSCAN(eps=5, min_samples=5, algorithm='kd_tree', n_jobs=-1)

    ndx = np.column_stack(np.where(test_image > 0.))

    ndx = np.append(ndx, np.expand_dims(test_image[test_image > 0.], axis=1), axis=1)

    try:
        # Perform (2D) clustering on the data and returns cluster labels the points are associated with
        c_labels = db_scan.fit_predict(ndx)
    except ValueError:
        return []

    if len(c_labels) < 1:
        return []

    clusters = get_clusters(ndx, c_labels)

    clusters = _validate_clusters(clusters)

    return clusters


# @profile
@timeit
def msds_q(test_image, true_tracks, noise_est, debug, profiler):
    limits = get_limits(test_image, true_tracks)

    pub = False
    ext = '.pdf'

    # limits = (0, 70, 2000, 2160)   #s limits for crossing streaks
    # limits = (50,120, 1100, 1400)
    # limits = None
    profiler.reset()

    ndx = pre_process_data(test_image)
    profiler.add_timing('MSDS: pre_process_data')

    # Build quad/nd tree that spans all the data points
    k_tree = build_tree(ndx, leave_size=40, n_axis=2)
    profiler.add_timing('MSDS: build_tree')

    visualise_filtered_data(ndx, true_tracks, '1_filtered_data' + ext, limits=limits, debug=debug, pub=pub)

    # Traverse the tree and identify valid linear streaks
    leaves = traverse(k_tree.tree, ndx, bbox=(0, test_image.shape[1], 0, test_image.shape[0]), min_length=2.,
                      noise_est=noise_est)
    profiler.add_timing('MSDS: traverse')

    positives = process_leaves(leaves)
    profiler.add_timing('MSDS: process_leaves')

    print("Processed {} leaves. Of which {} were positives.".format(len(leaves), len(positives)))

    visualise_tree_traversal(ndx, true_tracks, positives, leaves, '2_processed_leaves' + ext, limits=limits,
                             vis=False, pub=True)
    eps = estimate_leave_eps(positives)
    profiler.add_timing('MSDS: estimate_leave_eps')

    cluster_data = h_cluster_leaves(positives, distance_thold=eps)
    profiler.add_timing('MSDS: h_cluster_leaves')
    visualise_clusters(cluster_data, true_tracks, positives,
                       filename='3_clusters' + ext,
                       limits=limits,
                       debug=debug, pub=pub)

    # lp = LineProfiler()
    # lp_wrapper = lp(validate_clusters)
    # lp_wrapper(cluster_data)
    # lp.print_stats()

    # lp = LineProfiler()
    # lp_wrapper = lp(process_leaves)
    # lp_wrapper(leaves)
    # lp.print_stats()

    # Filter invalid clusters
    tracks = validate_clusters(cluster_data)
    profiler.add_timing('MSDS: validate_clusters')
    visualise_tracks(tracks, true_tracks, '4_tracks' + ext, limits=limits, debug=debug, pub=pub)

    visualise_tracks(tracks, true_tracks, '5_tracks' + ext, limits=None, debug=debug, pub=pub)
    return tracks


@timeit
def msds_2(test_image, true_tracks, noise_est, debug, profiler):
    limits = get_limits(test_image, true_tracks)

    pub = False
    ext = '.pdf'

    # limits = (0, 70, 2000, 2160)   #s limits for crossing streaks
    # limits = (50,120, 1100, 1400)
    # limits = None
    profiler.reset()

    ndx = msds2.pre_process_data(test_image)
    profiler.add_timing('MSDS: pre_process_data')

    # Build quad/nd tree that spans all the data points
    k_tree = msds2.build_tree(ndx, leave_size=40, n_axis=2)
    profiler.add_timing('MSDS: build_tree')

    visualise_filtered_data(ndx, true_tracks, '1_filtered_data' + ext, limits=limits, debug=debug, pub=pub)

    # Traverse the tree and identify valid linear streaks
    leaves = msds2.traverse(k_tree.tree, ndx, bbox=(0, test_image.shape[1], 0, test_image.shape[0]), min_length=2.,
                            noise_est=noise_est)
    profiler.add_timing('MSDS: traverse')

    # pre-preprocessing step to be included in tree traversal
    leave_data = np.zeros(shape=(len(leaves), 45, 3))

    for i, l in enumerate(leaves):
        leave_data[i, :len(l[0]), ...] = l[0]

    positives = msds2.process_leaves(leave_data)
    profiler.add_timing('MSDS: process_leaves')

    profiler.show()

    lp = LineProfiler()
    lp_wrapper = lp(msds2.process_leaves)
    lp_wrapper(leave_data)
    lp.print_stats()

    print("Processed {} leaves. Of which {} were positives.".format(len(leaves), len(positives)))

    visualise_tree_traversal(ndx, true_tracks, positives, leaves, '2_processed_leaves' + ext, limits=limits,
                             vis=False, pub=True)
    eps = msds2.estimate_leave_eps(positives)
    profiler.add_timing('MSDS: estimate_leave_eps')

    cluster_data = msds2.h_cluster_leaves(positives, distance_thold=eps)
    profiler.add_timing('MSDS: h_cluster_leaves')
    visualise_clusters(cluster_data, true_tracks, positives,
                       filename='3_clusters' + ext,
                       limits=limits,
                       debug=debug, pub=pub)

    # lp = LineProfiler()
    # lp_wrapper = lp(process_leaves)
    # lp_wrapper(leaves)
    # lp.print_stats()

    # Filter invalid clusters
    tracks = msds2.validate_clusters(cluster_data)
    profiler.add_timing('MSDS: validate_clusters')
    visualise_tracks(tracks, true_tracks, '4_tracks' + ext, limits=limits, debug=debug, pub=pub)

    visualise_tracks(tracks, true_tracks, '5_tracks' + ext, limits=None, debug=debug, pub=pub)
    return tracks
