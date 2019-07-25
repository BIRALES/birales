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
import seaborn as sns
from configuration import *
from evaluation import *
from filters import *
from msds import __inertia_ratio, traverse, mydist, __plot_leave, __plot_candidate, h_cluster, __plot_track, \
    __plot_clusters, h_cluster2, similar, cluster_merge, fill, fit, cluster_merge_pd, mydist2
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


def __best_group(fclust1, bb, i):
    u, c = np.unique(fclust1, return_counts=True)
    u_groups = u[c > 2]

    group_score = np.zeros(shape=(len(u_groups), 3))
    group_score[:, 0] = u_groups
    df1 = pd.DataFrame()
    for j, g in enumerate(u_groups):
        c = bb[np.where(fclust1 == g)]
        ratio, c2 = __inertia_ratio(c[:, 1], c[:, 0], c[:, 2], i, g)
        if i == 140:
            print i, ratio, len(c2)
            _viz_cluster(bb, fclust1)

        if ratio / len(c2) < 0.2:
            # print 'x'
            df1 = df1.append(c2)
    # if i == 1957:
    #     print i, len(df1), df1
    if len(df1) < 1:
        return np.nan, -1, []

    if len(df1['group'].unique()) < 1:
        return np.nan, -2, []

    valid = df1[df1['ratio_n'] < 0.1]
    if len(valid) < 1:
        return np.nan, -3, []

    # best = valid.loc[valid['ratio_n'].idxmax()]  # gets best groups
    best = valid.iloc[valid['ratio_n'].values.argmin()]

    best_group = np.reshape(best['group'], 1)
    best_ratio = best['ratio_n']

    # todo - merge best groups

    return best_ratio, best_group, valid


def __best_group2(fclust1, bb, i):
    u, c = np.unique(fclust1, return_counts=True)
    u_groups = u[c > 2]

    group_score = np.zeros(shape=(len(u_groups), 3))
    group_score[:, 0] = u_groups
    df1 = pd.DataFrame()
    for j, g in enumerate(u_groups):
        c = bb[np.where(fclust1 == g)]
        n = len(c)
        if n >= 20:
            ratio = 0.19
        elif len(np.unique(c[:, 0])) < 2 or len(np.unique(c[:, 1])) < 2:
            ratio = 0.21
        else:
            ratio, c = __inertia_ratio(c[:, 1], c[:, 0], i, g)

            df1.append(c)
        group_score[j, 1] = ratio

        group_score[j, 2] = ratio / n

    # best_groups = group_score[group_score[:, 1] < 0.25][:, 0]
    best_groups = group_score[group_score[:, 2] < 0.1]

    if len(best_groups) < 1:
        return np.nan, []

    best_ratio = best_groups[np.argmin(best_groups[:, 2])][1]
    best_group = np.reshape(best_groups[np.argmin(best_groups[:, 2])][0], 1)

    # if i in [1957]:
    #     print '\n', i
    #     _viz_cluster(bb, fclust1)
    #     print group_score
    #     print best_groups
    #     print best_ratio, best_group

    return best_ratio, best_group

    ## weighted check of ratio
    # best_ratio = np.min(best_groups[:, 1])
    # best_ratio = best_groups[np.argmin(best_groups[:, 2])][1]
    # best_group = np.reshape(best_groups[np.argmin(best_groups[:, 2])][0], 1)
    # if len(best_groups) > 1:
    #
    #     c = bb[np.in1d(fclust1, best_groups[:, 0])]
    #     ratio_combined = __inertia_ratio(c[:, 1], c[:, 0], i)
    #
    #     if ratio_combined < best_ratio:
    #         return ratio_combined, best_groups[:, 0]

    ## unweighted check
    best_ratio = best_groups[np.argmin(best_groups[:, 1])][1]
    best_group = np.reshape(best_groups[np.argmin(best_groups[:, 1])][0], 1)
    if len(best_groups) > 1:

        c = bb[np.in1d(fclust1, best_groups[:, 0])]
        ratio_combined = __inertia_ratio(c[:, 1], c[:, 0], i)

        if ratio_combined < best_ratio:
            return ratio_combined, best_groups[:, 0]

    return best_ratio, best_group


def msds_q(test_image):
    t1 = time.time()
    candidates = __msds_q(test_image, clustering_thres=2.0, vis=True)

    print 'MSDS found {} Clusters in {:0.2f} seconds'.format(len(candidates), time.time() - t1)

    return candidates


def _td(t1):
    return time.time() - t1


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

    t4 = time.time()
    r, leaves = traverse(ktree.tree, ndx, 0, 256, 0, 4096)
    print 'Tree traversed in {:0.3f} seconds. Found {} leaves'.format(_td(t4), len(leaves))

    fig, ax = plt.subplots(1)
    if vis:
        ax.plot(ndx[:, 1], ndx[:, 0], '.')

    clusters = []
    candidates = []
    clusters = pd.DataFrame()
    t5 = time.time()
    for i, leave in enumerate(leaves[1100:1400]):
        # i += 1900
        data, x1, x2, y1, y2, n = leave

        fclust1 = fclusterdata(data, 3.0, metric=mydist2, criterion='distance')
        ratio, best_gs, cluster = __best_group(fclust1, data, i)
        # print  ratio, best_gs, data
        if best_gs < 0:
            if vis:
                __plot_leave(ax, x1, y1, x2, y2, i, score='No G. R:{:0.3f}'.format(ratio), positive=False)
            continue

        if ratio == 0.2 or ratio == 0.:
            if vis:
                __plot_leave(ax, x1, y1, x2, y2, i, score='Ratio 101', positive=False)
            continue

        # cluster = data
        # noise = data[np.in1d(fclust1, best_gs, invert=True)]

        noise = None
        # ratio passed. Check it is long enough
        n = len(cluster)
        if n < 3:
            if vis:
                __plot_leave(ax, x1, y1, x2, y2, i, score='N < 3', positive=False, cluster=cluster, noise=noise)
            continue

        # Now test for correlation

        if len(np.unique(cluster['sample'])) == 1:
            pear = [1., 0.05]
        elif len(np.unique(cluster['channel'])) == 1:
            pear = [1., 0.05]
        else:
            pear = pearsonr(cluster['sample'], cluster['channel'])

        if pear[1] > 0.05 and n <= 20:
            # if pear[0] > -0.80 or pear[1] > 0.05:
            if vis:
                d = n / (max(cluster['channel']) - min(cluster['channel'])) * (
                        max(cluster['sample']) - min(cluster['sample']))
                score = 'Low C:{:0.3f}\nS:{:0.3f}\nN:{}\nD:{:0.3f}'.format(pear[0], pear[1], n, d)
                __plot_leave(ax, x1, y1, x2, y2, i, score=score, positive=False, cluster=cluster, noise=noise)
            continue

        # streak is correlated AND with a low inertia -> add to the list of possible clusters
        # cluster = np.append(cluster, np.expand_dims(np.full(shape=n, fill_value=i), axis=1), axis=1)
        cluster['leave_id'] = i
        clusters = clusters.append(cluster)

        if vis:
            positive = True
            m, c, r_value, _, _ = linregress(cluster['sample'], cluster['channel'])
            score = 'I:{:0.3f}\nM:{:0.3f}\nC:{:0.3f}\nR:{:0.3f}'.format(ratio, m, c, r_value)
            __plot_leave(ax, x1, y1, x2, y2, i, score=score, positive=positive, cluster=cluster, noise=noise)

    print '{} Clusters generated in {:0.3f} seconds'.format(len(clusters), time.time() - t5)

    t3 = time.time()
    # cc = np.vstack(clusters)
    df = clusters
    ax.set_ylim(1117, 1178)
    ax.set_xlim(75, 231)
    plt.show()

    # df = pd.DataFrame({'channel': cc[:, 0], 'sample': cc[:, 1], 'snr': cc[:, 2], 'group': cc[:, 3]})

    fclust1, u_groups = h_cluster2(X=df, threshold=20.)
    df['group_2'] = fclust1

    print '2nd Pass Clustering finished in {:0.2f} seconds. Found {} groups.'.format(time.time() - t3, len(u_groups))
    plt.clf()
    for g in u_groups:
        data = df[df['group_2'] == g]
        ax = sns.scatterplot(data=data, x='sample', y='channel')
        ax.annotate(g, (np.mean(data['sample']), np.mean(data['channel'])))
    ax.figure.savefig("before_ransac.png")

    # ransac = linear_model.RANSACRegressor(residual_threshold=10)
    t1 = time.time()
    plt.clf()
    for g in u_groups:
        c = df[df['group_2'] == g]
        leave_id = c['leave_id'].iloc[0]
        if len(c) < 2:
            continue

        m, intercept, r_value, p, e = linregress(c['sample'], c['channel'])

        ratio, c2 = __inertia_ratio(c['sample'], c['channel'], c['snr'], g)
        print g, 'R:{:0.5f} P:{:0.5f} E:{:0.5f} I:{:0.5f}'.format(r_value, p, e, ratio)
        # if p < 0.05 and e < 0.1 and ratio < 0.021:
        if p < 0.05 and e < 0.1 and ratio < 0.3:
            print g, 'R:{:0.5f} P:{:0.5f} E:{:0.5f} I:{:0.5f}'.format(r_value, p, e, ratio)
            # e = fill(test_img, cluster, ransac)
            ax = sns.scatterplot(c['sample'], c['channel'])
            ax.annotate(g, (np.mean(c['sample']), np.mean(c['channel'])))
            candidates.append(c)
    ax.figure.savefig("after_ransac_all.png")
    print '{} Groups reduced to {} candidates after filtering 1. Time taken: {:0.3f} seconds'.format(len(u_groups),
                                                                                                     len(candidates),
                                                                                                     _td(t1))

    t3 = time.time()
    ptracks = []
    for j, candidate in enumerate(candidates):
        for i, track in enumerate(ptracks):
            # If beam candidate is similar to candidate, merge it.
            if similar(track, candidate, i, j):
                c = cluster_merge_pd(track, candidate)

                ransac.fit(c['channel'].values.reshape(-1, 1), c['sample'])
                candidate, valid = fit(c, ransac.inlier_mask_)

                ptracks[i] = fill(test_img, candidate, ransac)

                break
        else:
            ransac.fit(candidate['channel'].values.reshape(-1, 1), candidate['sample'])
            candidate, valid = fit(candidate, ransac.inlier_mask_)

            candidate = fill(test_img, candidate, ransac)

            ptracks.append(candidate)

    print '{} candidates merged into {} candidates after similarity check. Time taken: {:0.3f} seconds'.format(
        len(candidates),
        len(ptracks),
        _td(t3))

    t4 = time.time()
    c2 = []
    for i, c in enumerate(ptracks):
        m, intercept, r_value, p, e = linregress(c['sample'], c['channel'])
        ratio, c3 = __inertia_ratio(c['sample'], c['channel'], c['snr'], g)
        print i, 'R:{:0.5f} P:{:0.5f} E:{:0.5f} I:{:0.5f}'.format(r_value, p, e, ratio)
        if r_value < -0.95 and e < 0.01 and ratio < 0.20:
            c2.append(c3)

    print '{} candidates merged into {} candidates after last sanity check. Time taken: {:0.3f} seconds'.format(
        len(ptracks),
        len(c2),
        _td(t4))

    plt.clf()
    for i, c in enumerate(c2):
        m, intercept, r_value, p, e = linregress(c['sample'], c['channel'])

        ratio, c4 = __inertia_ratio(c['sample'], c['channel'], c['snr'], i)
        x = c['sample'].mean()
        y = c['channel'].mean()

        print i, 'R:{:0.5f} P:{:0.5f} E:{:0.5f} I:{:0.5f}'.format(r_value, p, e, ratio)
        ax.annotate(i, (x, y))
        ax = sns.scatterplot(c['sample'], c['channel'], marker=".")

    ax.figure.savefig("after_ransac_all_merged.png")

    for i, (track_x, track_y) in enumerate(tracks):
        ax.plot(track_x, track_y, 'o', color='k', zorder=-1)
    ax.figure.savefig("after_ransac_all_merged_with_tracks.png")

    return c2


if __name__ == '__main__':

    # snr = [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    # snr = [2, 55]
    snr = [5]
    N_TRACKS = 2
    N_CHANS = 4096
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
