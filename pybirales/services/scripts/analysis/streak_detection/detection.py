"""
This script lists the feature algorithm that will be used to extract
space debris tracks from the filtered data

"""

import pandas as pd
from astride import Streak
from astride.utils.edge import EDGE
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import KDTree
from scipy.stats import linregress, pearsonr
from skimage import measure
from skimage.transform import probabilistic_hough_line
from sklearn import linear_model
from sklearn.cluster import DBSCAN

from configuration import *
from evaluation import *
from filters import *
from msds import __inertia_ratio, traverse, mydist, __plot_leave, __plot_candidates
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
                shape_cut=0.5, area_cut=10,
                radius_dev_cut=0.5,
                connectivity_angle=3.)
    edge.quantify()

    # Filter the edges, so only streak remains.
    # edge.filter_edges()
    edge.connect_edges()

    streaks = edge.get_edges()

    clusters = []
    for i, streak in enumerate(streaks):
        cluster = np.column_stack([streak['y'], streak['x'], np.full(len(streak['x']), i)])
        clusters.append(cluster)

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


def msds_c(test_image, unfiltered_image=None):
    """
    Multi-pixel streak detection strategy

    Algorithm combines the beam data across the multi-pixel in order to increase SNR whilst
    reducing the computational load. Data is transformed such that data points belonging
    to the same streak, cluster around a common point.  Then, a noise-aware clustering algorithm,
    such as DBSCAN, can be applied on the data points to identify the candidate tracks.

    :return:
    """

    db_scan = DBSCAN(eps=3, min_samples=5, algorithm='kd_tree', n_jobs=-1)

    ndx = np.column_stack(np.where(test_image > 0.))

    ndx = np.append(ndx, np.expand_dims(test_image[test_image > 0.], axis=1), axis=1)

    # Coordinate transform into rho, theta
    # theta
    t = np.arctan(ndx[:, 1] / (4096 - ndx[:, 0]))

    # rho
    r = ndx[:, 0] / np.cos(t)

    tr = np.column_stack((t, r, test_image[test_image > 0.]))

    ndx = np.append(ndx, np.expand_dims(t, axis=1), axis=1)

    # ndx = np.append(ndx, np.expand_dims(r, axis=1), axis=1)

    # ndx = np.append(ndx, np.expand_dims(t, axis=1), axis=1)

    # plt.plot(ndx[:,1], t, '.')
    # plt.show()

    # DBSCAN and generate clusters
    try:
        # Perform (2D) clustering on the data and returns cluster labels the points are associated with
        c_labels = db_scan.fit_predict(ndx)
    except ValueError:
        return []

    if len(c_labels) < 1:
        return []

    # Add cluster labels to the data
    labelled_data = np.append(ndx, np.expand_dims(c_labels, axis=1), axis=1)

    # Cluster mask to remove noise clusters
    de_noised_data = labelled_data[labelled_data[:, 3] > -1]

    de_noised_data = de_noised_data[de_noised_data[:, 3].argsort()]

    # Return the location at which clusters where identified
    cluster_ids = np.unique(de_noised_data[:, 3], return_index=True)

    # Split the data into clusters
    clusters = np.split(de_noised_data, cluster_ids[1])

    # remove empty clusters
    clusters = [x for x in clusters if np.any(x)]

    return clusters

    # Convert back the data points


def msds_q(test_image):
    # Coordinate transform channel, time coordinates into rho, theta

    # Build quad/nd tree that spans all the data points
    ndx = np.column_stack(np.where(test_image > 0.))
    ndx = np.append(ndx, np.expand_dims(test_image[test_image > 0.], axis=1), axis=1)

    t1 = time.time()
    ktree = KDTree(ndx, 30)
    print 'Tree built in ', time.time() - t1, ' seconds'

    t1 = time.time()
    r, leaves = traverse(ktree.tree, 0, 256, 0, 4096)
    print 'Tree traversed in ', time.time() - t1, ' seconds'

    fig, ax = plt.subplots(1)
    ax.plot(ndx[:, 1], ndx[:, 0], '.')

    clusters = []
    t1 = time.time()
    n_leaves = len(leaves)
    candidates = []
    for i, rectangle in enumerate(leaves):
        a, b, x1, x2, y1, y2, n = rectangle

        ys = ndx[:, 0]

        aa = ndx[np.logical_and(ys >= y1, ys <= y2)]
        bb = aa[np.logical_and(aa[:, 1] >= x1, aa[:, 1] <= x2)]

        if not np.any(bb):
            continue

        fclust1 = fclusterdata(bb, 5.0, metric=mydist, criterion='distance')

        best = None
        best_r = 0
        u, c = np.unique(fclust1, return_counts=True)
        u_groups = u[c > 2]

        # s_best = None
        best_gs = []
        for g in u_groups:
            c = bb[np.where(fclust1 == g)]
            pear = pearsonr(c[:, 0], c[:, 1])
            # if i == 364:
            #     print g, len(c), pear[0], pear[1]

            if pear[0] < best_r and pear[1] < 0.05 and len(c) > 2:
                # s_best = best
                best_r = pear[0]
                best = g

            if pear[0] < -0.85 and pear[1] < 0.05 and len(c) > 2:
                best_gs.append(g)

        if len(best_gs) > 1:
            tmp_cluster = bb[np.in1d(fclust1, best_gs)]
            tmp_noise = bb[np.in1d(fclust1, best_gs, invert=True)]
            pear = pearsonr(tmp_cluster[:, 0], tmp_cluster[:, 1])
            if pear[0] < best_r and pear[1] < 0.05:
                cluster = tmp_cluster
                noise = tmp_noise
            else:
                continue
        else:
            cluster = bb[np.where(fclust1 == best)]
            noise = bb[np.where(fclust1 != best)]

        n = len(cluster[:, 0])
        if n < 3:
            __plot_leave(ax, x1, y1, x2, y2, i, score='N < 3', positive=False)
            continue

        ratio = __inertia_ratio(cluster[:, 1], cluster[:, 0], i)
        condition = ratio < 0.2 and len(cluster[:, 1]) > 3
        m, c, r_value, _, _ = linregress(cluster[:, 0], cluster[:, 1])

        score = 'I:{:0.3f}\nM:{:0.3f}\nC:{:0.3f}\nR:{:0.3f}'.format(ratio, m, c, r_value)

        if condition:
            __plot_leave(ax, x1, y1, x2, y2, i, score=score, positive=True, cluster=cluster, noise=noise)
            cluster = np.append(cluster, np.expand_dims(np.full(shape=len(cluster), fill_value=i), axis=1), axis=1)

            clusters.append(cluster)

        __plot_leave(ax, x1, y1, x2, y2, i, score=score, positive=False)

    cc = np.vstack(clusters)
    fclust1 = fclusterdata(cc, 100.0, metric=mydist, criterion='distance')

    u, c = np.unique(fclust1, return_counts=True)
    u_groups = u[c > 2]
    ransac = linear_model.RANSACRegressor(residual_threshold=5)
    t2 = time.time()
    for g in u_groups:
        c = cc[np.where(fclust1 == g)]

        ransac.fit(c[:, 0].reshape(-1, 1), c[:, 1])
        inlier_mask = ransac.inlier_mask_
        # outlier_mask = np.logical_not(inlier_mask)

        d = c[inlier_mask]

        pear = pearsonr(d[:, 0], d[:, 1])
        if pear[0] < -0.9 and pear[1] < 0.05 and len(d) > 10:
            # print g, len(d), len(c), pear[0], pear[1]
            X = np.arange(min(d[:, 0]), max(d[:, 0]))
            missing = np.setxor1d(X, d[:, 0])
            if len(missing) > len(d):
                # candidates.append(d)
                continue

            missing_y = np.setxor1d(np.arange(min(d[:, 1]), max(d[:, 1])), d[:, 1])
            missing_x = (missing_y - ransac.estimator_.intercept_)/ransac.estimator_.coef_[0]
            missing_x = missing_x.astype(int)

            y = ransac.predict(missing.reshape(-1, 1))

            y = np.concatenate([y, missing_y])
            missing = np.concatenate([missing, missing_x])


            m = np.column_stack((missing, y)).astype(int)

            p  = test_image[m[:, 0], m[:, 1]]

            m = np.append(m, np.expand_dims(p, axis=1), axis=1)
            m = np.append(m, np.expand_dims(np.full(shape=len(m), fill_value=g), axis=1), axis=1)

            e = np.concatenate([d, m], axis=0)
            candidates.append(e)

            # print g, len(e), min(e[:, 0]), max(e[:, 0]), min(e[:, 1]), max(e[:, 1])

            # print 'added ', len(m), ' data points to cluster', g
        else:
            # print 'dropped cluster', g, 'of length', len(c), pear[0], pear[1]
            continue

    n_clusters = len(clusters)
    print 'found {} clusters from {} leaves in {:0.3f} seconds'.format(n_clusters, n_leaves, time.time() - t1)

    print 'Merged {} clusters into {} candidates'.format(n_clusters, len(candidates), time.time() - t2)

    __plot_candidates(ax, candidates)

    # ax.set_ylim(-5, 70)
    # ax.set_xlim(-5, 70)
    # ax.set_ylim(1085, 1209)
    # ax.set_xlim(0, 256)


    ax.set_ylim(1114.0, 1179.0)
    ax.set_xlim(72.0, 232.0)

    plt.show()
    print 'Clustering finished in ', time.time() - t1, ' seconds'
    return candidates


if __name__ == '__main__':

    # snr = [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    # snr = [2, 55]
    snr = [5]
    N_TRACKS = 10
    N_CHANS = 4096
    N_SAMPLES = 256
    metrics = {}
    metrics_df = pd.DataFrame()

    detectors = [
        # ('Naive DBSCAN', naive_dbscan, ('Filter', sigma_clipping)),
        # ('HDBSCAN', hdbscan, ('Filter', sigma_clipping)),
        ('msds_q', msds_q, ('Filter', sigma_clipping)),
        # ('Hough Transform', hough_transform, ('Filter', global_thres)),
        # ('Astride', astride, ('No Filter', no_filter)),
        ('Old Astride', astride_old, ('No Filter', no_filter)),
        # ('CFAR', None, ('No Filter', cfar)),
    ]

    # Create image from real data
    test_img = create_test_img(os.path.join(ROOT, FITS_FILE), nchans=N_CHANS, nsamples=N_SAMPLES)

    # Remove channels with RFI
    test_img = rfi_filter(test_img)
    # visualise_image(test_img, 'Test Image: no tracks', tracks=None)

    # Generate a number of tracks
    tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_LENGTH_RANGE, test_img.shape, TRACK_THICKNESS)

    # Estimate the noise
    noise_mean = np.mean(test_img)

    for d_name, detect, (f_name, filter_func) in detectors:
        # filtering_algorithm = ('Filter', filter_func)
        for s in snr:
            metrics[s] = {}
            metrics_tmp_df = pd.DataFrame()
            print "\nEvaluating filters with tracks at SNR {:0.2f}W".format(s)

            # Add tracks to the true data
            true_image = add_tracks(np.zeros(shape=test_img.shape), tracks, noise_mean, s)

            data = test_img.copy()

            # Add tracks to the simulated data
            data = add_tracks(data, tracks, noise_mean, s)

            # visualise_image(data, 'Test Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)

            # visualise_image(true_image, 'Truth Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)

            if filter_func != no_filter:
                # Filter the data in generate metrics
                # data = test_img.copy()

                # Split data in chunks
                start = time.time()
                mask, threshold = chunked_filtering(data, filter_func)
                mask, _ = hit_and_miss(data, test_img, mask)
                timing = time.time() - start
                print "The {} filtering algorithm, finished in {:2.3f}s".format(f_name, timing)

                visualise_filter(data, mask, tracks, f_name, s, threshold, visualise=False)

                metrics[s][f_name] = evaluate_filter(true_image, data, ~mask, timing, s, TRACK_THICKNESS)

                # data[:] = False
                # data[~mask] = True
                data[mask] = -100

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

    print metrics_df[['f1', 'sensitivity', 'specificity', 'reduction', 'dt', 'score']].sort_values(by=['score'], )
