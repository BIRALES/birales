"""
This script lists the feature algorithm that will be used to extract
space debris tracks from the filtered data

"""

from functools import partial

from astride import Streak
from astride.utils.edge import EDGE
from scipy.spatial import KDTree
from scipy.stats import linregress, pearsonr
from skimage import measure
from skimage.transform import probabilistic_hough_line
from sklearn import linear_model
from sklearn.cluster import DBSCAN

from configuration import *
from evaluation import *
from filters import *
from msds import traverse, __plot_leave, h_cluster2, h_cluster3, similar, fill, fit, cluster_merge_pd, __ir
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


# profile = line_profiler.LineProfiler()


# @profile


def msds_q(test_image):
    t1 = time.time()
    candidates = __msds_q(test_image, clustering_thres=2.0, vis=False)

    print 'MSDS found {} Clusters in {:0.2f} seconds'.format(len(candidates), time.time() - t1)

    return candidates


def _td(t1):
    return time.time() - t1


def split(test_img, ransac, candidate, candidates, ptracks, g):
    g1 = g
    if np.sum(~ransac.inlier_mask_) > 0.2 * len(candidate):

        print 'Candidate {} will be split'.format(g1)
        split_candidate_1 = candidate[ransac.inlier_mask_]
        split_candidate_2 = candidate[~ransac.inlier_mask_]
        ransac.fit(split_candidate_1[:, 0].reshape(-1, 1), split_candidate_1[:, 1])
        candidate_1, valid_1 = fit(split_candidate_1, ransac.inlier_mask_)

        if valid_1:
            # candidate_1 = fill(test_img, candidate_1, ransac)
            print 'Sub-candidate 1/{} added to candidates'.format(g1)
            candidates.append(candidate_1)

        ransac.fit(split_candidate_2[:, 0].reshape(-1, 1), split_candidate_2[:, 1])
        candidates_2, valid_2 = fit(split_candidate_2, ransac.inlier_mask_)

        if valid_2:
            # candidates_2 = fill(test_img, candidates_2, ransac)
            print 'Sub-candidate 2/{} added to candidates'.format(g1)

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
            print 'Candidate {} added to tracks'.format(g1)
        else:
            print 'Candidate {} dropped'.format(g1)

    return candidates, ptracks


# @profile
def __msds_q(test_image, clustering_thres=2.0, vis=False):
    # Coordinate transform channel, time coordinates into rho, theta

    ransac = linear_model.RANSACRegressor(residual_threshold=5)

    # Build quad/nd tree that spans all the data points
    t0 = time.time()
    ndx = np.column_stack(np.where(test_image > 0.))
    ndx = np.append(ndx, np.expand_dims(test_image[test_image > 0.], axis=1), axis=1)
    print 'Data prepared in {:0.3f} seconds'.format(_td(t0))

    t1 = time.time()
    ktree = KDTree(ndx, 30)
    print 'Tree built in {:0.3f} seconds'.format(_td(t1))

    if vis or debug:
        fig, ax = plt.subplots(1)
        ax.plot(ndx[:, 1], ndx[:, 0], '.', 'r')

    t4 = time.time()
    rectangles, leaves = traverse(ktree.tree, ndx, 0, test_image.shape[1], 0, test_image.shape[0])
    print 'Tree traversed in {:0.3f} seconds. Found {} leaves'.format(_td(t4), len(leaves))

    # func = partial(get_leaves, test_image)
    # leaves = pool.map(func, [0 ,1])

    candidates = []
    t5 = time.time()
    clusters = [(cluster, i, x, y) for i, (cluster, best_gs, ratio, x1, x2, y1, y2, n, x, y) in enumerate(leaves)]
    # clusters = [cluster for (cluster, best_gs, ratio, x1, x2, y1, y2, n, x, y) in leaves]
    print '{} Clusters generated in {:0.3f} seconds'.format(len(clusters), time.time() - t5)

    if vis:
        for i, (cluster, rejected, best_gs, msg, x1, x2, y1, y2, n) in enumerate(rectangles):
            __plot_leave(ax, x1, y1, x2, y2, i, msg, False, positives=cluster, negatives=rejected)

        for i, (cluster, best_gs, msg, x1, x2, y1, y2, n, _, _) in enumerate(leaves):
            __plot_leave(ax, x1, y1, x2, y2, i, msg, True, positives=cluster, negatives=None)

        # ax.set_ylim(50, 200)
        # ax.set_xlim(30, 166)
        # plt.show()

    t3 = time.time()

    df = np.vstack(clusters)

    fclust1, u_groups = h_cluster3(X=df, threshold=20.)

    # df = np.append(df, np.expand_dims(fclust1, axis=1), axis=1)

    print '2nd Pass Clustering finished in {:0.2f} seconds. Found {} groups.'.format(time.time() - t3, len(u_groups))

    if debug:
        plt.clf()
        for g in u_groups:
            c = np.vstack(df[fclust1 == g, 0])
            ax = sns.scatterplot(x=c[:, 1], y=c[:, 0])
            ax.annotate(g, (np.mean(c[:, 1]), np.mean(c[:, 0])))

        for i, (track_x, track_y) in enumerate(tracks):
            ax.plot(track_x, track_y, 'o', color='k', zorder=-1)

        ax.figure.savefig("clusters.png")

    t1 = time.time()
    for g in u_groups:
        # c = df[df[:, 3] == g]
        c = np.vstack(df[fclust1 == g, 0])

        m, intercept, r_value, p, e = linregress(c[:, 1], c[:, 0])

        if p < 0.05 and e < 0.1 and len(c) > 4:
            c = np.append(c, np.expand_dims(np.full(len(c), g), axis=1), axis=1)
            candidates.append(c)

    print '{} Groups reduced to {} candidates after filtering 1. Time taken: {:0.3f} seconds'.format(len(u_groups),
                                                                                                     len(candidates),
                                                                                                     _td(t1))

    if debug:
        plt.clf()

        for c in candidates:
            group_2 = c[:, 3][0]
            ax = sns.scatterplot(x=c[:, 1], y=c[:, 0])
            ax.annotate(group_2, (np.mean(c[:, 1]), np.mean(c[:, 0])))

        for i, (track_x, track_y) in enumerate(tracks):
            ax.plot(track_x, track_y, 'o', color='k', zorder=-1)
        ax.figure.savefig("clusters_2.png")

    t3 = time.time()
    ptracks = []
    for j, candidate in enumerate(candidates):
        g = candidate[:, 3][0]
        for i, track in enumerate(ptracks):
            # If beam candidate is similar to candidate, merge it.
            # print 'Comparing candidate {} with track {}'.format(g, i)
            if similar(track, candidate, i, j):
                # print 'Candidate {} and track {} are similar'.format(g, i)
                c = cluster_merge_pd(track, candidate)

                ransac.fit(c[:, 0].reshape(-1, 1), c[:, 1])
                candidate, valid = fit(c, ransac.inlier_mask_)

                # candidates, ptracks = split(ransac, c, candidates, ptracks)

                # ptracks[i] = fill(test_img, candidate, ransac)
                ptracks[i] = candidate
                break
        else:
            # print 'Candidate {} is unique'.format(g)
            ransac.fit(candidate[:, 0].reshape(-1, 1), candidate[:, 1])

            candidates, ptracks = split(test_img, ransac, candidate, candidates, ptracks, g)

    print '{} candidates merged into {} candidates after similarity check. Time taken: {:0.3f} seconds'.format(
        len(candidates),
        len(ptracks),
        _td(t3))

    if debug:
        plt.clf()

        for c in ptracks:
            group_2 = c[:, 3][0]
            ax = sns.scatterplot(x=c[:, 1], y=c[:, 0])
            ax.annotate(group_2, (np.mean(c[:, 1]), np.mean(c[:, 0])))

        for i, (track_x, track_y) in enumerate(tracks):
            ax.plot(track_x, track_y, 'o', color='k', zorder=-1)
        ax.figure.savefig("clusters_3.png")

    t4 = time.time()
    c2 = []
    for i, c in enumerate(ptracks):
        m, intercept, r_value, p, e = linregress(c[:, 1], c[:, 0])
        ratio = __ir(c[:, :2], g=None)

        missing_channels = np.setxor1d(np.arange(min(c[:, 0]), max(c[:, 0])),
                                       c[:, 0])

        if len(missing_channels) / len(c) > 0.45:
            continue

        time_span = len(np.unique(c[:, 1]))
        print i, c[:, 3][0], 'R:{:0.5f} P:{:0.5f} E:{:0.5f} I:{:0.5f} N:{}'.format(r_value, p, e, ratio, time_span)
        if r_value < -0.95 and ratio < 0.20 and time_span > 15:
            ransac = ransac.fit(c[:, 0].reshape(-1, 1), c[:, 1])
            c3 = fill(test_img, c, ransac)
            c2.append(c3)

            # print len(c2), len(missing_channels), len(c)

    print '{} candidates merged into {} candidates after last sanity check. Time taken: {:0.3f} seconds'.format(
        len(ptracks),
        len(c2),
        _td(t4))

    if debug:
        plt.clf()

        for i, c in enumerate(c2):
            m, intercept, r_value, p, e = linregress(c[:, 1], c[:, 0])

            ratio = __ir(c[:, :2], i)
            x = c[:, 1].mean()
            y = c[:, 0].mean()

            print i + 1, 'R:{:0.5f} P:{:0.5f} E:{:0.5f} I:{:0.5f} N:{}'.format(r_value, p, e, ratio, len(c))

            missing = c[c[:, 3] == -2]
            thickened = c[c[:, 3] == -3]
            detected = c[c[:, 3] >= 0]
            ax = sns.scatterplot(detected[:, 1], detected[:, 0], color='green', marker=".", zorder=4)
            ax = sns.scatterplot(thickened[:, 1], thickened[:, 0], marker=".", color='pink', zorder=2, edgecolor="k")
            ax = sns.scatterplot(missing[:, 1], missing[:, 0], marker="+", color='red', zorder=3, edgecolor="k")
            # ax.set_ylim(360, 384)
            # ax.set_xlim(28, 156)
            ax.annotate('Group {}'.format(i + 1), (x, 1.01 * y), zorder=3)

        for i, (track_x, track_y) in enumerate(tracks):
            ax.plot(track_x, track_y, 'o', color='k', zorder=-1)
        ax.figure.savefig("clusters_4.png")

    return c2


if __name__ == '__main__':
    debug = False
    vis = False
    # snr = [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    # snr = [2, 55]
    snr = [0.5, 1, 2, 3, 5, 8, 13, 21]
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

    # Remove channels with RFI
    test_img = rfi_filter(test_img)
    # visualise_image(test_img, 'Test Image: no tracks', tracks=None)

    # Reduce the input problem to speed up computation - use for debugging only
    # test_img = test_img[0: 2200, :]

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

                visualise_filter(data, mask, tracks, f_name, s, threshold, visualise=False)

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

    print metrics_df[['snr', 'sensitivity', 'specificity', 'dt', 'precision', 'recall']].sort_values(by=['precision'], )
