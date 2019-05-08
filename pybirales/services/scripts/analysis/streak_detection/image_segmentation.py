"""
This script evaluates the performance of different filters for
image segmentation of the spectrogram produced by the PyBirales
Channeliser.

Notebook reproduces the figures used in the
streak detection paper.

"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from skimage.filters import threshold_otsu, threshold_local
from sklearn.metrics import jaccard_similarity_score, f1_score, precision_score, recall_score, accuracy_score, \
    mean_squared_error
from skimage.measure import compare_ssim as ssim
import pandas as pd
from skimage import img_as_float

plt.rcParams['figure.figsize'] = (12, 10)


def save_figure(filename):
    if SAVE_FIGURES:
        plt.savefig(os.path.join(OUT_DIR, OBS_NAME, filename, '.pdf'))


def calculate_amplitude(noise_avg_db, snr):
    """

    :param noise_std:
    :param snr:
    :return:
    """
    # noise_avg_watts = noise_std ** 2
    # noise_avg_db = 10 * np.log10(noise_avg_watts)
    #
    # signal_avg_db = noise_avg_db + snr
    # sig_avg_watts = 10 ** (signal_avg_db / 10)
    #
    # # we can use this as amplitude for our chirp
    # return np.sqrt(sig_avg_watts)

    signal_power = noise_avg_db + snr

    return signal_power


def visualise_image(image, title):
    """
    Visualise the test data
    :param image:
    :return:
    """
    if VISUALISE:
        ax = sns.heatmap(image, cbar_kws={'label': 'Power (dB)'}, xticklabels=25, yticklabels=25)
        ax.invert_yaxis()
        ax.set(xlabel='Time sample', ylabel='Channel', title=title)

        plt.show()


def create_track(x, gradient, intercept, img_shape):
    y = x * gradient + intercept
    y[y <= 0] = 0
    y[y >= img_shape[0]] = img_shape[0] - 1
    return y.astype(int)


def create_test_img(filepath, nchans=192, nsamples=120):
    """

    :param filepath:
    :param scaling_factor: downscale factor
    :return:
    """
    fits_file = fits.open(filepath)
    size = np.shape(fits_file[0].data[0])

    n_chans = size[0] - nchans
    n_samples = size[1] - nsamples

    test_img = fits_file[0].data[0][n_chans:, n_samples:]
    print 'Test Image of size {} was generated'.format(test_img.shape)

    return test_img


def get_test_tracks(n_tracks, gradient, track_length, image_shape, noise_mean, snr):
    tracks = []
    for i in range(0, n_tracks):
        m = np.random.uniform(gradient[0], gradient[1])
        start = np.random.randint(low=0, high=image_shape[1])
        end = np.random.randint(low=start + track_length[0], high=start + track_length[1])
        end = np.amin([end, image_shape[1]])
        x = np.arange(start, end)
        c = np.random.randint(low=100, high=image_shape[0])
        y = create_track(x, m, c, image_shape)
        # print start, end, start + np.random.randint(low=track_length[0], high=track_length[1]), track_length
        print 'Created track with m={:0.2f}, c={:0.1f}, of {}px at ({},{}) with SNR={:0.2f}dB'.format(m, c, (
                max(x) - start), start, max(y), snr)

        tracks.append((x, y, calculate_amplitude(noise_mean, snr=snr)))
    return tracks


def add_tracks(image, tracks):
    for track in tracks:
        image[track[1], track[0]] += track[2]
    return image


def visualise_filter(data, mask, threshold, filename=None):
    if VISUALISE:
        ax = sns.heatmap(data, cbar_kws={'label': 'Power (dB)'},
                         xticklabels=25,
                         yticklabels=25,
                         mask=mask)
        ax.invert_yaxis()
        ax.set(xlabel='Time sample', ylabel='Channel', title=filename)

        print('{}: Noise: {:0.2f} dB, Thres: {:0.2f} dB (~SNR: {:0.2f} dB)'.format(filename, np.mean(data),
                                                                                   np.mean(threshold),
                                                                                   np.mean(threshold / np.mean(data))))
        plt.show()


def evaluate_filter(truth_img, test_img, positives, exec_time, snr):
    """

    :param truth:
    :param prediction:
    :param positives: Location where the filter thinks there should be a target
    :return:
    """


    truth = truth_img.ravel()
    truth[truth > 0.] = True
    truth[truth <= 0.] = False

    test_img[:] = False
    test_img[positives] = True
    prediction = test_img.ravel()

    ssim_score = ssim((truth).astype('float64'), (prediction).astype('float64'))

    return {
        # 'jaccard': jaccard_similarity_score(truth, prediction),
        'f1': f1_score(truth, prediction),
        'precision': precision_score(truth, prediction),
        'recall': recall_score(truth, prediction),
        'accuracy': accuracy_score(truth, prediction),
        # 'mse': mean_squared_error(truth, prediction),
        'ssim': ssim_score,
        # 'timing': exec_time,
        # 'nchans': truth_img.shape[0],
        # 'nsamples': truth_img.shape[1],
        'snr': snr
    }


def global_thres(test_img, true_img, snr):
    # BF Global Filter
    start = time.time()
    channel_noise = np.mean(test_img)
    std = np.std(test_img)

    threshold = 2 * std + channel_noise

    global_threshold_mask = test_img < threshold
    end = time.time()
    visualise_filter(test_img, global_threshold_mask, threshold, 'Global Threshold Filter at %d dB' % snr)

    return evaluate_filter(true_img, test_img, ~global_threshold_mask, end - start, s)


def local_thres(test_img, true_img, snr):
    # BF Local Filter
    start = time.time()
    channel_noise = np.mean(test_img, axis=1)
    std = np.std(test_img, axis=1)

    threshold = 2 * std + channel_noise

    local_threshold_mask = test_img < np.expand_dims(threshold, axis=1)
    end = time.time()
    visualise_filter(test_img, local_threshold_mask, threshold, 'Local Threshold Filter at %d dB' % snr)

    return evaluate_filter(true_img, test_img, ~local_threshold_mask, end - start, s)


def otsu_thres(test_img, true_img, snr):
    # Otsu Filter
    start = time.time()
    global_thresh = threshold_otsu(test_img)
    global_filter_mask = test_img < global_thresh
    visualise_filter(test_img, global_filter_mask, global_thresh, 'Global Otsu filter at %d dB' % snr)
    end = time.time()
    return evaluate_filter(true_img, test_img, ~global_filter_mask, end - start, s)


def adaptive(test_img, true_img, snr):
    # Adaptive Threshold filter
    start = time.time()
    block_size = 5
    local_thresh = threshold_local(test_img, block_size, offset=2)
    local_filter_mask = test_img < local_thresh
    end = time.time()
    visualise_filter(test_img, local_filter_mask, local_thresh, 'Local Adaptive Filter at %d dB' % snr)

    return evaluate_filter(true_img, test_img, ~local_filter_mask, end - start, s)


def median(test_img, true_img, snr):
    pass


# Plot the metrics
# Metrics as a function of SNR
# Filters as a function of speed

if __name__ == '__main__':

    ROOT = "/home/denis/.birales/visualisation/fits"
    OUT_DIR = "/home/denis/.birales/visualisation/analysis"
    SAVE_FIGURES = False
    TX = 410.085e6
    BNDW = 0.078125e6
    START_FREQ = 410.0425e6
    END_FREQ = 410.0425e6 + BNDW
    N_CHANS = 8192
    SAMPLING_RATE = 78125
    OBS_NAME = "NORAD_1328"
    SNR = 20
    N_TRACKS = 5
    TD = 262144 / 78125 / 32.
    CD = 78125 / 8192.
    F = (1. / TD) / (1. / CD)
    GRADIENT_RANGE = np.array([-0.57, -50.47]) / F
    SNR_RANGE = [5, 50]
    TRACK_LENGTH_RANGE = np.array([5, 15]) / TD  # in seconds
    # TRACK_START = np.array([0, 10]) / TD  # in seconds
    FITS_FILE = "norad_1328/norad_1328_raw_0.fits"
    VISUALISE = False
    SEED = 56789
    np.random.seed(SEED)

    # snr = [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    # snr = [2, 55]
    snr = [30]
    metrics = {}
    metrics_df = pd.DataFrame()
    for s in snr:
        metrics_tmp_df = pd.DataFrame()
        print "\nEvaluating filters with tracks at {:0.2f} dB".format(s)
        # Create image from real data
        test_img = create_test_img(os.path.join(ROOT, FITS_FILE), nchans=2000, nsamples=1400)

        # Estimate the noise
        NOISE_MEAN = np.mean(test_img)

        # Generate a number of tracks
        tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_LENGTH_RANGE, test_img.shape, NOISE_MEAN, s)

        # Add tracks to the true data
        true_image = add_tracks(np.zeros(shape=test_img.shape), tracks)

        # Add tracks to the simulated data
        test_img = add_tracks(test_img, tracks)

        visualise_image(test_img, 'Test Image: %d tracks at %d dB' % (N_TRACKS, s))

        visualise_image(true_image, 'Truth Image: %d tracks at %d dB' % (N_TRACKS, s))

        # Filter the data in generate metrics
        metrics[s] = {
            'Global threshold': global_thres(test_img.copy(), true_image, s),
            'Local threshold': local_thres(test_img.copy(), true_image, s),
            'Otsu threshold': otsu_thres(test_img.copy(), true_image, s),
            'Adaptive threshold': adaptive(test_img.copy(), true_image, s),
            # 'Median Filter': median(test_img.copy(), true_image, s),
        }
        metrics_tmp_df = metrics_tmp_df.from_dict(metrics[s], orient='index')
        metrics_df = metrics_df.append(metrics_tmp_df)

    print metrics_df
