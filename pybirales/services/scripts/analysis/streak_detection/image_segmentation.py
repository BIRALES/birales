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


def calculate_amplitude(noise_avg, snr):
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

    # return noise_avg_db + snr
    print noise_avg, snr
    return 10 ** (snr / 10) + noise_avg


def visualise_image(image, title, tracks=None):
    """
    Visualise the test data
    :param image:
    :return:
    """
    if VISUALISE:
        x_start, x_end, y_start, y_end = get_limits(image, tracks)

        prev_shape = image.shape

        print "Visualising a subset of the image {} from {}".format(image[y_start:y_end, x_start:x_end].shape,
                                                                    prev_shape)

        ax = sns.heatmap(image[y_start:y_end, x_start:x_end], cbar_kws={'label': 'Power (dB)'}, xticklabels=25,
                         yticklabels=25)
        ax.invert_yaxis()
        # ax.set(ylim=(y_start, y_end), xlim=(x_start, x_end))

        ax.set(xlabel='Time sample', ylabel='Channel', title=title)

        plt.show()


def get_limits(image, tracks):
    x_start = 0
    x_end = image.shape[1]

    y_start = 0
    y_end = image.shape[0]

    # print x_start, x_end, y_start, y_end
    if tracks:
        m_track = tracks[0]
        for t in tracks:
            if len(t[0]) > len(m_track[0]):
                m_track = t

        x_m = np.mean(m_track[0]).astype(int)
        y_m = np.mean(m_track[1]).astype(int)

        x_range = np.max(m_track[0]) - np.min(m_track[0])
        y_range = np.max(m_track[1]) - np.min(m_track[1])
        x_start = np.amax([x_m - x_range, 0])
        x_end = np.amin([x_m + x_range, x_end])

        y_start = np.amax([y_m - y_range, 0])
        y_end = np.amin([y_m + y_range, y_end])

    # print x_start, x_end, y_start, y_end

    return x_start, x_end, y_start, y_end


def create_track(x, gradient, intercept, img_shape):
    def thicken(x, y, t):
        # x_new = np.tile(x, t).ravel()
        # y_new = np.tile(y, t)
        x_new = np.array([x for _ in range(t)])
        y_new = np.array([y + i for i in range(t)])

        # length = x.shape[0]\
        # chunks = [[0, length]
        #
        # ]
        #
        # for i, chunk in enumerate(chunks):
        #     tmp = np.take(y_new, chunk)
        #     tmp += i

        return x_new, y_new

    y = x * gradient + intercept

    x, y = thicken(x, y, 5)

    y[y <= 0] = 0
    y[y >= img_shape[0]] = img_shape[0] - 1

    x[x <= 0] = 0
    x[x >= img_shape[1]] = img_shape[1] - 1

    return np.ravel(x.astype(int)), np.ravel(y.astype(int))


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

    # return np.random.normal(0, 0.5, (nchans, nsamples))
    return test_img


def get_test_tracks(n_tracks, gradient, track_length, image_shape):
    tracks = []

    for i in range(0, n_tracks):
        m = np.random.uniform(gradient[0], gradient[1])
        start = np.random.randint(low=0, high=image_shape[1] / 3.)
        end = np.random.randint(low=start + track_length[0], high=start + track_length[1])
        end = np.amin([end, image_shape[1]])
        x = np.arange(start, end)
        c = np.random.randint(low=100, high=image_shape[0])
        x, y = create_track(x, m, c, image_shape)

        print 'Created track with m={:0.2f}, c={:0.1f}, of {}px at ({},{}) to ({},{})'.format(m, c, (
                max(x) - start), start, max(y), end, min(y))

        tracks.append((x, y))
    return tracks


def add_tracks(image, tracks, noise_mean, snr):
    power = calculate_amplitude(noise_mean, snr=snr)

    for track in tracks:
        image[track[1], track[0]] += power
    return image


def visualise_filter(data, mask, filter_str, tracks, filename=None):
    if VISUALISE:
        x_start, x_end, y_start, y_end = get_limits(data, tracks)
        ax = sns.heatmap(data[y_start:y_end, x_start:x_end], cbar_kws={'label': 'Power (dB)'},
                         xticklabels=25,
                         yticklabels=25,
                         mask=mask[y_start:y_end, x_start:x_end])
        # ax.set(ylim=(y_start, y_end), xlim=(x_start, x_end))
        ax.invert_yaxis()
        ax.set(xlabel='Time sample', ylabel='Channel', title=filter_str)
        plt.show()

    print filter_str


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
        'precision': precision_score(truth, prediction, average='binary'),
        'recall': recall_score(truth, prediction),
        'accuracy': accuracy_score(truth, prediction),
        # 'mse': mean_squared_error(truth, prediction),
        'ssim': ssim_score,
        # 'timing': exec_time,
        # 'nchans': truth_img.shape[0],
        # 'nsamples': truth_img.shape[1],
        'snr': snr
    }


def global_thres(test_img, true_img, tracks, snr):
    """
    BF Global Filter
    :param test_img:
    :param true_img:
    :param snr:
    :return:
    """

    start = time.time()
    channel_noise = np.mean(test_img)
    std = np.std(test_img)

    threshold = 3 * std + channel_noise

    global_threshold_mask = test_img < threshold
    end = time.time()

    filter_str = 'Global Threshold Filter at SNR {} dB. Threshold at {:2.2f} dB'.format(snr, threshold)
    visualise_filter(test_img, global_threshold_mask, filter_str, tracks, 'global_threshold')

    return evaluate_filter(true_img, test_img, ~global_threshold_mask, end - start, s)


def local_thres(test_img, true_img, tracks, snr):
    """
    BF Local Filter
    :param test_img:
    :param true_img:
    :param snr:
    :return:
    """

    start = time.time()
    channel_noise = np.mean(test_img, axis=1)
    std = np.std(test_img, axis=1)

    threshold = 3 * std + channel_noise

    local_threshold_mask = test_img < np.expand_dims(threshold, axis=1)
    end = time.time()

    filter_str = 'Local Threshold Filter at SNR {} dB. Threshold at {:2.2f} dB'.format(snr, np.mean(threshold))
    visualise_filter(test_img, local_threshold_mask, filter_str, tracks, 'local_threshold')

    return evaluate_filter(true_img, test_img, ~local_threshold_mask, end - start, s)


def otsu_thres(test_img, true_img, tracks, snr):
    """
    Otsu Filter
    :param test_img:
    :param true_img:
    :param snr:
    :return:
    """

    start = time.time()
    global_thresh = threshold_otsu(test_img)
    global_filter_mask = test_img < global_thresh
    end = time.time()

    filter_str = 'Global Otsu filter at SNR {} dB. Threshold at {:2.2f} dB'.format(snr, global_thresh)
    visualise_filter(test_img, global_filter_mask, filter_str, tracks, 'global_otsu')

    return evaluate_filter(true_img, test_img, ~global_filter_mask, end - start, s)


def adaptive(test_img, true_img, tracks, snr):
    """
    Adaptive Threshold filter
    :param test_img:
    :param true_img:
    :param snr:
    :return:
    """

    start = time.time()
    block_size = 5
    local_thresh = threshold_local(test_img, block_size, offset=2)
    local_filter_mask = test_img < local_thresh
    end = time.time()
    # print local_thresh
    filter_str = 'Local Adaptive Filter at SNR {} dB. Threshold at {:2.2f} dB'.format(snr, np.mean(local_thresh))
    visualise_filter(test_img, local_filter_mask, filter_str, tracks, 'local_adaptive')

    return evaluate_filter(true_img, test_img, ~local_filter_mask, end - start, s)


def median(test_img, true_img, tracks, snr):
    """

    :param test_img:
    :param true_img:
    :param tracks:
    :param snr:
    :return:
    """
    pass


# Plot the metrics
# Metrics as a function of SNR
# Filters as a function of speed

if __name__ == '__main__':

    ROOT = "/home/denis/.birales/visualisation/fits"
    OUT_DIR = "/home/denis/.birales/visualisation/analysis"
    SAVE_FIGURES = False
    OBS_NAME = "NORAD_1328"
    N_TRACKS = 50
    TD = 262144 / 78125 / 32.
    CD = 78125 / 8192.
    F = (1. / TD) / (1. / CD)
    GRADIENT_RANGE = np.array([-0.57, -50.47]) / F
    TRACK_LENGTH_RANGE = np.array([5, 15]) / TD  # in seconds
    # FITS_FILE = "norad_1328/norad_1328_raw_0.fits"
    FITS_FILE = "filter_test/filter_test_raw_0.fits"
    VISUALISE = True
    SEED = 56789
    np.random.seed(SEED)

    # snr = [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    # snr = [2, 55]
    snr = [2]
    metrics = {}
    metrics_df = pd.DataFrame()
    for s in snr:
        metrics_tmp_df = pd.DataFrame()
        print "\nEvaluating filters with tracks at SNR {:0.2f} dB".format(s)
        # Create image from real data
        test_img = create_test_img(os.path.join(ROOT, FITS_FILE), nchans=8192, nsamples=32*10)
        # visualise_image(test_img, 'Test Image: no tracks', tracks=None)

        # Estimate the noise
        noise_mean = np.mean(test_img)

        # Generate a number of tracks
        tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_LENGTH_RANGE, test_img.shape)

        # Add tracks to the true data
        true_image = add_tracks(np.zeros(shape=test_img.shape), tracks, noise_mean, s)

        # Add tracks to the simulated data
        test_img = add_tracks(test_img, tracks, noise_mean, s)

        visualise_image(test_img, 'Test Image: %d tracks at SNR %d dB' % (N_TRACKS, s), tracks)

        visualise_image(true_image, 'Truth Image: %d tracks at SNR %d dB' % (N_TRACKS, s), tracks)

        # Filter the data in generate metrics
        metrics[s] = {
            'Global threshold': global_thres(test_img.copy(), true_image, tracks, s),
            'Local threshold': local_thres(test_img.copy(), true_image, tracks, s),
            'Otsu threshold': otsu_thres(test_img.copy(), true_image, tracks, s),
            'Adaptive threshold': adaptive(test_img.copy(), true_image, tracks, s),
            # 'Median Filter': median(test_img.copy(), true_image, s),
        }
        metrics_tmp_df = metrics_tmp_df.from_dict(metrics[s], orient='index')
        metrics_df = metrics_df.append(metrics_tmp_df)

    print metrics_df
