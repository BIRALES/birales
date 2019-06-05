"""
This script lists the feature algorithm that will be used to extract
space debris tracks from the filtered data

"""

import os
import time

import pandas as pd
from sklearn.cluster import DBSCAN

from configuration import *
from evaluation import *
from filters import *
from receiver import *
from visualisation import *


def hough_transform(test_image):
    """
    Hough feature detection

    Clustering within the dataset are identified using the Hough transform

    :return:
    """
    pass


def astride(test_image):
    """
    ASTRIDE feature detection

    Clustering within the dataset are identified using the ASTRIDE streak detection algorithm

    :return:
    """
    pass


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

    # Add cluster labels to the data
    labelled_data = np.append(ndx, np.expand_dims(c_labels, axis=1), axis=1)

    # Cluster mask to remove noise clusters
    de_noised_data = labelled_data[labelled_data[:, 3] > -1]

    de_noised_data = de_noised_data[de_noised_data[:, 3].argsort()]

    # Return the location at which clusters where identified
    cluster_ids = np.unique(de_noised_data[:, 3], return_index=True)

    # Split the data into clusters
    clusters = np.split(de_noised_data, cluster_ids[1])

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
    snr = [5]
    metrics = {}
    metrics_df = pd.DataFrame()

    detectors = [
        ('Naive DBSCAN', naive_dbscan),
        # ('Hough Transform', hough_transform),
        # ('MSDS', msds),
        # ('Astride', astride),
    ]

    filtering_algorithm = ('Global Filter', global_thres)

    for s in snr:
        metrics[s] = {}

    for s in snr:
        metrics_tmp_df = pd.DataFrame()
        print "\nEvaluating filters with tracks at SNR {:0.2f}W".format(s)
        # Create image from real data
        test_img = create_test_img(os.path.join(ROOT, FITS_FILE), nchans=4096, nsamples=256)

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

        visualise_image(test_img, 'Test Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)

        # visualise_image(true_image, 'Truth Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)

        # Filter the data in generate metrics
        f_name, filter_func = filtering_algorithm
        data = test_img.copy()

        # split data in chunks
        start = time.time()
        mask, threshold = chunked_filtering(data, filter_func)
        mask, _ = hit_and_miss(data, test_img, mask)
        timing = time.time() - start

        # Visualise filter output
        metrics[s][f_name] = evaluate_filter(true_image, data, ~mask, timing, s, TRACK_THICKNESS)
        visualise_filter(test_img, mask, tracks, f_name, s, threshold, visualise=False)

        metrics_tmp_df = metrics_tmp_df.from_dict(metrics[s], orient='index')
        metrics_df = metrics_df.append(metrics_tmp_df)

        # feature extraction - detection
        for d_name, d in detectors:
            candidates = d(data)

            print "The {} algorithm, found {} candidates".format(d_name, len(candidates))

            # visualise candidates

            # evaluation

        # data association

    print metrics_df.sort_values(by=['score'], ascending=False)
