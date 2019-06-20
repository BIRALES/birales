"""
This script lists the feature algorithm that will be used to extract
space debris tracks from the filtered data

"""

import time

import pandas as pd
from sklearn.cluster import DBSCAN

from configuration import *
from evaluation import *
from filters import *
from receiver import *
from visualisation import *
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line, radon, iradon
import matplotlib.pyplot as plt
from skimage.feature import canny
from astride import Streak


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


def naive_dbscan(test_image, unfiltered_image=None):
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


def msds(test_image):
    """
    Multi-pixel streak detection strategy

    Algorithm combines the beam data across the multi-pixel in order to increase SNR whilst
    reducing the computational load. Data is transformed such that data points belonging
    to the same streak, cluster around a common point.  Then, a noise-aware clustering algorithm,
    such as DBSCAN, can be applied on the data points to identify the candidate tracks.

    :return:
    """
    pass


if __name__ == '__main__':

    # snr = [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    # snr = [2, 55]
    snr = [3]
    metrics = {}
    metrics_df = pd.DataFrame()

    detectors = [
        # ('Naive DBSCAN', naive_dbscan, global_thres),
        # ('Hough Transform', hough_transform, global_thres),
        # ('MSDS', msds, global_thres),
        ('Astride', astride, ('No Filter', no_filter)),
        # ('CFAR', None, ('No Filter', cfar)),
    ]

    for d_name, detect, (f_name, filter_func) in detectors:
        # filtering_algorithm = ('Filter', filter_func)
        for s in snr:
            metrics[s] = {}
            metrics_tmp_df = pd.DataFrame()
            print "\nEvaluating filters with tracks at SNR {:0.2f}W".format(s)
            # Create image from real data
            test_img = create_test_img(os.path.join(ROOT, FITS_FILE), nchans=4096, nsamples=512)

            # Remove channels with RFI
            test_img = rfi_filter(test_img)

            # visualise_image(test_img, 'Test Image: no tracks', tracks=None)

            # Estimate the noise
            noise_mean = np.mean(test_img)

            # Generate a number of tracks
            tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_LENGTH_RANGE, test_img.shape, TRACK_THICKNESS)

            # Add tracks to the true data
            true_image = add_tracks(np.zeros(shape=test_img.shape), tracks, noise_mean, s)

            # Add tracks to the simulated data
            test_img = add_tracks(test_img, tracks, noise_mean, s)

            # visualise_image(test_img, 'Test Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)

            # visualise_image(true_image, 'Truth Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)

            data = test_img.copy()
            if filter_func != no_filter:
                # Filter the data in generate metrics
                data = test_img.copy()

                # Split data in chunks
                start = time.time()
                mask, threshold = chunked_filtering(data, filter_func)
                mask, _ = hit_and_miss(data, test_img, mask)
                timing = time.time() - start
                print "The {} filtering algorithm, finished in {:2.3f}s".format(f_name, timing)

                visualise_filter(test_img, mask, tracks, f_name, s, threshold, visualise=False)
                metrics[s][f_name] = evaluate_filter(true_image, data, ~mask, timing, s, TRACK_THICKNESS)

                data[mask] = -100

            # feature extraction - detection
            print "Running {} algorithm".format(d_name)
            start = time.time()
            candidates = detect(data)
            timing = time.time() - start
            print "The {} detection algorithm, found {} candidates in {:2.3f}s".format(d_name, len(candidates), timing)

            # visualise candidates
            visualise_detector(data, candidates, tracks, d_name, s, visualise=True)

            # evaluation
            metrics[s][d_name] = evaluate_detector(true_image, test_img, candidates, timing, s, TRACK_THICKNESS)

            metrics_df = metrics_df.append(pd.DataFrame.from_dict(metrics[s], orient='index'))
        # data association

    print metrics_df[['f1', 'sensitivity', 'specificity', 'reduction', 'dt', 'score']].sort_values(by=['score'],
                                                                                                   ascending=False)

    # for s in snr:
    #     metrics[s] = {}
    #
    # for s in snr:
    #     metrics_tmp_df = pd.DataFrame()
    #     print "\nEvaluating filters with tracks at SNR {:0.2f}W".format(s)
    #     # Create image from real data
    #     test_img = create_test_img(os.path.join(ROOT, FITS_FILE), nchans=4096, nsamples=512)
    #
    #     # Remove channels with RFI
    #     test_img = rfi_filter(test_img)
    #
    #     # visualise_image(test_img, 'Test Image: no tracks', tracks=None)
    #
    #     # Estimate the noise
    #     noise_mean = np.mean(test_img)
    #
    #     # Generate a number of tracks
    #     tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_LENGTH_RANGE, test_img.shape, TRACK_THICKNESS)
    #
    #     # Add tracks to the true data
    #     true_image = add_tracks(np.zeros(shape=test_img.shape), tracks, noise_mean, s)
    #
    #     # Add tracks to the simulated data
    #     test_img = add_tracks(test_img, tracks, noise_mean, s)
    #
    #     # visualise_image(test_img, 'Test Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)
    #
    #     # visualise_image(true_image, 'Truth Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)
    #
    #     # Filter the data in generate metrics
    #     f_name, filter_func = filtering_algorithm
    #     data = test_img.copy()
    #
    #     # split data in chunks
    #     start = time.time()
    #     mask, threshold = chunked_filtering(data, filter_func)
    #     mask, _ = hit_and_miss(data, test_img, mask)
    #     timing = time.time() - start
    #     print "The {} filtering algorithm, finished in {:2.3f}s".format(f_name, timing)
    #
    #     visualise_filter(test_img, mask, tracks, f_name, s, threshold, visualise=False)
    #     metrics[s][f_name] = evaluate_filter(true_image, data, ~mask, timing, s, TRACK_THICKNESS)
    #     # feature extraction - detection
    #     for d_name, detect, _ in detectors:
    #         print "Running {} algorithm".format(d_name)
    #         start = time.time()
    #         data[mask] = -100
    #         candidates = detect(test_img)
    #         timing = time.time() - start
    #         print "The {} algorithm, found {} candidates in {:2.3f}s".format(d_name, len(candidates), timing)
    #
    #         # visualise candidates
    #         visualise_detector(data, candidates, tracks, d_name, s, visualise=True)
    #
    #         # evaluation
    #         metrics[s][d_name] = evaluate_detector(true_image, test_img, candidates, timing, s, TRACK_THICKNESS)
    #
    #     metrics_df = metrics_df.append(pd.DataFrame.from_dict(metrics[s], orient='index'))
    #     # data association
    #
    # print metrics_df[['f1', 'sensitivity', 'specificity', 'reduction', 'dt', 'score']].sort_values(by=['score'], ascending=False)
