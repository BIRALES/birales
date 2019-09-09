"""
This script lists the feature algorithm that will be used to extract
space debris tracks from the filtered data

"""

from astride import Streak
from astride.utils.edge import EDGE
from skimage import measure
from skimage.transform import probabilistic_hough_line
from sklearn.cluster import DBSCAN

from configuration import *
from evaluation import *
from filters import *
from msds import traverse, cluster_leaves, pre_process_data, \
    build_tree, validate_clusters2, fill_clusters2, m_process_leaves, process_leaves
from receiver import *
from util import get_clusters
from visualisation import *


def hough_transform(test_image, unfiltered_image=None):
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

    return clusters


def astride(test_image):
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
        # cluster = np.column_stack([streak['y'], streak['x'], np.full(len(streak['x']), i)])

        cluster_df = pd.DataFrame(columns=['channel', 'sample', 'snr'])
        cluster_df['channel'] = streak['y']
        cluster_df['sample'] = streak['x']
        cluster_df['snr'] = 1

        clusters.append(cluster_df)

    return clusters


def astride_old(test_image):
    """
    ASTRIDE feature detection

    Clustering within the dataset are identified using the ASTRIDE streak detection algorithm

    :return:
    """

    # Create a temporary fits file from the raw, unfiltered data
    tmp_filepath = '/tmp/birales/unfiltered.fits'
    if not os.path.exists(os.path.dirname(tmp_filepath)):
        os.makedirs(os.path.dirname(tmp_filepath))

    header = fits.Header()
    header.set('OBS', 'TMP_data')
    fits.writeto(tmp_filepath, test_image, overwrite=True, header=header)

    streak = Streak(tmp_filepath, remove_bkg='constant', min_points=10, shape_cut=0.5, connectivity_angle=3)

    streak.detect()

    streaks = streak.streaks

    clusters = []
    for i, streak in enumerate(streaks):
        n = len(streak['y'])
        cluster = np.column_stack([np.array(streak['y']), np.array(streak['x']), np.full(n, 1), np.full(n, i)])
        clusters.append(cluster)

    return clusters


def hdbscan(test_image):
    from hdbscan import HDBSCAN
    def sim(u, v):
        dx = v[0] - u[0]
        dy = v[1] - u[1]
        t = np.arctan(dy / dx)
        return t
        if 0.5 * np.pi <= t <= np.pi and 1.5 * np.pi <= t <= 2 * np.pi:
            return dy / np.sin(t)
        return 1.
        # print x,y
        # return 1

    hdbscan = HDBSCAN(algorithm='prims_kdtree', alpha=1.0, approx_min_span_tree=True,
                      gen_min_span_tree=False, leaf_size=50, metric='euclidean',
                      min_cluster_size=5, min_samples=None, p=None)
    ndx = np.column_stack(np.where(test_image > 0.))

    ndx = np.append(ndx, np.expand_dims(test_image[test_image > 0.], axis=1), axis=1)

    # ndx = pairwise_distances(ndx, metric=sim)

    c_labels = hdbscan.fit_predict(ndx)

    clusters = get_clusters(ndx, c_labels)

    return clusters


def naive_dbscan(test_image):
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

    return clusters


def _viz_cluster(bb, fclust1):
    for b, g in zip(bb, fclust1):
        print b[0], b[1], b[2], g


# @profile
def msds_q(test_image):
    limits = get_limits(test_image, true_tracks)
    limits = (50, 150, 6000, 6500)
    # limits = (40, 200, 500, 1000)
    limits = (50, 175, 4000, 7100)
    limits = None

    # Pre-process the input data
    ndx = pre_process_data(test_image, noise_estimate=0.93711)

    # Build quad/nd tree that spans all the data points
    k_tree = build_tree(ndx, leave_size=30, n_axis=2, true_tracks=true_tracks, limits=limits, debug=debug,
                        visualisation=vis)

    # Traverse the tree and identify valid linear streaks
    rectangles, leaves = traverse(k_tree.tree, ndx,
                                  bbox=(0, test_image.shape[1], 0, test_image.shape[0]),
                                  distance_thold=3., min_length=2., cluster_size_thold=10.)

    rejected, clusters, thres = process_leaves(leaves, distance_thold=3., cluster_size_thold=10.,
                                               ndx=ndx,
                                               true_tracks=true_tracks, limits=limits,
                                               visualisation=vis)

    # Cluster the leaves based on their vicinity to each other
    cluster_labels, unique_labels, cluster_data = cluster_leaves(clusters, distance_thold=thres,
                                                                 true_tracks=true_tracks, limits=limits,
                                                                 visualisation=debug)

    # Filter invalid clusters
    valid_clusters = validate_clusters2(cluster_data, labelled_clusters=cluster_labels, unique_labels=unique_labels,
                                        e_thold=0.2, true_tracks=true_tracks, limits=limits, visualisation=debug)

    # Fill any missing data (increase recall)
    valid_tracks = fill_clusters2(valid_clusters, test_image, true_tracks=true_tracks, visualisation=debug)

    return valid_tracks


if __name__ == '__main__':
    debug = False
    vis = False
    # snr = [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    # snr = [25]
    snr = [2, 3, 5, 10, 15, 20, 25]
    # snr = [3]
    N_TRACKS = 15
    N_CHANS = 8192
    N_SAMPLES = 256
    TRACK_THICKNESS = 1
    metrics = {}
    metrics_df = pd.DataFrame()

    detectors = [
        # ('Naive DBSCAN', naive_dbscan, ('Filter', sigma_clipping)),
        # ('HDBSCAN', hdbscan, ('Filter', sigma_clipping)),
        ('msds_q', msds_q, ('Filter', sigma_clipping)),
        # ('Hough Transform', hough_transform, ('Filter', global_thres)),
        # ('Astride', astride, ('No Filter', no_filter)),
        # ('Old Astride', astride_old, ('No Filter', no_filter)),
        # ('CFAR', None, ('No Filter', cfar)),
    ]

    # Create image from real data
    test_img = create_test_img(os.path.join(ROOT, FITS_FILE), nchans=N_CHANS, nsamples=N_SAMPLES)

    # Reduce the input problem to speed up computation - use for debugging only
    # test_img = test_img[0: 1000, :]

    # Remove channels with RFI
    filtered_test_img = rfi_filter(test_img)
    # visualise_image(test_img, 'Test Image: no tracks', tracks=None)

    # Generate a number of tracks
    true_tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_LENGTH_RANGE, filtered_test_img.shape,
                                  TRACK_THICKNESS)

    # Estimate the noise
    noise_mean = np.mean(test_img)

    print 'Mean power (noise): ', noise_mean

    for d_name, detect, (f_name, filter_func) in detectors:
        # filtering_algorithm = ('Filter', filter_func)
        for s in snr:
            metrics[s] = {}
            metrics_tmp_df = pd.DataFrame()
            np.random.seed(SEED)
            print "\nEvaluating filters with tracks at SNR={:0.2f} dB".format(s)

            # Add tracks to the true data
            true_image = add_tracks(np.zeros(shape=filtered_test_img.shape), true_tracks, noise_mean, s)

            data = filtered_test_img.copy()

            # Add tracks to the simulated data
            data = add_tracks(data, true_tracks, noise_mean, s)

            # visualise_image(data, 'Test Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks, visualise=True )

            # visualise_image(true_image, 'Truth Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)

            if filter_func != no_filter:
                # Filter the data in generate metrics
                # data = test_img.copy()

                # Split data in chunks
                start = time.time()
                mask, threshold = chunked_filtering(data, filter_func)
                # mask, _ = hit_and_miss(data, test_img, mask)
                timing = time.time() - start
                print "The {} filtering algorithm, finished in {:2.3f}s".format(f_name, timing)

                visualise_filter(data, mask, true_tracks, f_name, s, threshold, visualise=False)

                # metrics[s][f_name] = evaluate_filter(true_image, data, ~mask, timing, s, TRACK_THICKNESS)

                # data[:] = False
                # data[~mask] = True
                data[mask] = -100

                # print np.mean(mask), s, threshold
                # continue
            # continue
            # feature extraction - detection
            print "Running {} algorithm".format(d_name)
            start = time.time()
            candidates = detect(data)
            timing = time.time() - start
            print "The {} detection algorithm, found {} candidates in {:2.3f}s".format(d_name, len(candidates), timing)

            # visualise candidates
            # visualise_detector(data, candidates, tracks, d_name, s, visualise=True)

            # evaluation
            metrics[s][d_name] = evaluate_detector(true_image, data, candidates, timing, s, TRACK_THICKNESS)

            metrics_df = metrics_df.append(pd.DataFrame.from_dict(metrics[s], orient='index'))
        # data association

    print metrics_df[['snr', 'f1', 'N', 'dt', 'precision', 'recall']].sort_values(by=['precision'], )
