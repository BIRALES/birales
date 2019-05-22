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
import pandas as pd
import seaborn as sns
from astropy.io import fits
from scipy.ndimage import binary_hit_or_miss, binary_opening
from skimage.filters import *
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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

    # return 10 ** (snr / 10) + noise_avg

    return 10 ** (snr / 10. - np.log10(noise_avg))


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
        # a[np.nonzero(a)].mean()
        a = image[y_start:y_end, x_start:x_end]
        # print "mean:", a[np.nonzero(a)].max()

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

    # return 800, 875, 4190, 4268
    return x_start, x_end, y_start, y_end


def create_track(x, gradient, intercept, img_shape):
    def thicken(x, y, t):
        x_new = np.array([x for _ in range(t)])
        y_new = np.array([y + i for i in range(t)])
        return x_new, y_new

    y = x * gradient + intercept

    x, y = thicken(x, y, TRACK_THICKNESS)

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

    # n_chans = size[0] - nchans
    # n_samples = size[1] - nsamples

    test_img = fits_file[0].data[0][:nchans, :nsamples]
    print 'Test Image of size {} was generated. Noise estimate at {}W'.format(test_img.shape, np.mean(test_img))

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

        # print 'Created track with m={:0.2f}, c={:0.1f}, of {}px at ({},{}) to ({},{})'.format(m, c, (
        #         max(x) - start), start, max(y), end, min(y))

        tracks.append((x, y))
    print 'Created {} tracks'.format(n_tracks)
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
        # 'dt': exec_time,
        # 'nchans': truth_img.shape[0],
        # 'nsamples': truth_img.shape[1],
        'snr': snr,
        'n_px': np.sum(prediction).astype(np.int),
        'reduction':  (1 - (np.sum(prediction) / np.prod(truth_img.shape)) ) * 100.
    }


def global_thres(test_img, tracks, snr):
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

    threshold = 4 * std + channel_noise
    # threshold = 4.34

    global_threshold_mask = test_img < threshold
    end = time.time()

    filter_str = 'Global Threshold Filter at SNR {} dB. Threshold at {:2.2f}W'.format(snr, threshold)
    visualise_filter(test_img, global_threshold_mask, filter_str, tracks, 'global_threshold')

    return global_threshold_mask, end - start


def local_thres(test_img, tracks, snr):
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

    threshold = 4 * std + channel_noise

    local_threshold_mask = test_img < np.expand_dims(threshold, axis=1)
    end = time.time()

    filter_str = 'Local Threshold Filter at SNR {} dB. Threshold at {:2.2f}W'.format(snr, np.mean(threshold))
    visualise_filter(test_img, local_threshold_mask, filter_str, tracks, 'local_threshold')

    return local_threshold_mask, end - start


def otsu_thres(test_img, tracks, snr):
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

    filter_str = 'Global Otsu filter at SNR {} dB. Threshold at {:2.2f}W'.format(snr, global_thresh)
    visualise_filter(test_img, global_filter_mask, filter_str, tracks, 'global_otsu')

    return global_filter_mask, end - start


def adaptive(test_img, tracks, snr):
    """
    Adaptive Threshold filter
    :param test_img:
    :param true_img:
    :param snr:
    :return:
    """

    start = time.time()
    block_size = 151
    local_thresh = threshold_local(test_img, block_size=block_size)
    local_filter_mask = test_img < local_thresh
    end = time.time()

    filter_str = 'Local Adaptive Filter at SNR {} dB. Threshold at {:2.2f}W'.format(snr, np.mean(local_thresh))
    visualise_filter(test_img, local_filter_mask, filter_str, tracks, 'local_adaptive')

    return local_filter_mask, end - start


def yen(test_img, tracks, snr):
    start = time.time()

    local_thresh = threshold_yen(test_img, nbins=1024)
    local_filter_mask = test_img < local_thresh
    end = time.time()
    # print local_thresh
    filter_str = 'Yen Filter at SNR {} dB. Threshold at {:2.2f}W'.format(snr, np.mean(local_thresh))
    visualise_filter(test_img, local_filter_mask, filter_str, tracks, 'local_adaptive')

    return local_filter_mask, end - start


def minimum(test_img, tracks, snr):
    start = time.time()
    local_thresh = threshold_minimum(test_img, nbins=1024)
    local_filter_mask = test_img < local_thresh
    end = time.time()
    filter_str = 'Minimum Filter at SNR {} dB. Threshold at {:2.2f}W'.format(snr, np.mean(local_thresh))
    visualise_filter(test_img, local_filter_mask, filter_str, tracks, 'minimum')

    return local_filter_mask, end - start


def triangle(test_img, tracks, snr):
    start = time.time()
    local_thresh = threshold_triangle(test_img, nbins=1024)
    local_filter_mask = test_img < local_thresh
    end = time.time()
    filter_str = 'threshold_triangle at SNR {} dB. Threshold at {:2.2f}W'.format(snr, np.mean(local_thresh))
    visualise_filter(test_img, local_filter_mask, filter_str, tracks, 'threshold_triangle')

    return local_filter_mask, end - start


def isodata(test_img, tracks, snr):
    start = time.time()
    local_thresh = threshold_isodata(test_img, nbins=1024)
    local_filter_mask = test_img < local_thresh
    end = time.time()
    filter_str = 'ISOData Threshold at SNR {} dB. Threshold at {:2.2f}W'.format(snr, np.mean(local_thresh))
    visualise_filter(test_img, local_filter_mask, filter_str, tracks, 'threshold_isodata')

    return local_filter_mask, end - start


def hit_and_miss(data, test_img, f_name, mask):
    structure = np.zeros((5, 5))
    structure[2, 2] = 1

    start = time.time()
    mask += binary_hit_or_miss(data, structure1=structure)
    end = time.time()

    data[~mask] = test_img[~mask]
    visualise_filter(data, mask, 'Hit and Miss filter after ' + f_name, tracks, 'hit_and_miss')

    return mask, end - start


def morph_opening(data, test_img, f_name, mask):
    start = time.time()
    tmp = binary_opening(data, structure=np.ones((2, 2))).astype(np.int)
    mask += tmp < 1
    end = time.time()

    data[~mask] = test_img[~mask]

    visualise_filter(data, mask, 'M Opening remove filter after ' + f_name, tracks, 'morph_opening')

    return mask, end - start


def kittler(test_img, tracks, snr):
    start = time.time()
    h, g = np.histogram(test_img.ravel(), bins=1024)
    h = h.astype(np.float) +0.00001
    g = g.astype(np.float)+0.0001

    g = g[:-1]
    c = np.cumsum(h)
    m = np.cumsum(h * g)
    s = np.cumsum(h * g ** 2)
    sigma_f = np.sqrt(s / c - (m / c) ** 2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    sigma_b = np.sqrt(sb / cb - (mb / cb) ** 2)
    p = c / c[-1]
    v = p * np.log(sigma_f) + (1 - p) * np.log(sigma_b) - p * np.log(p) - (1 - p) * np.log(1 - p)
    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)
    t = g[idx]

    mask = data <= t

    end = time.time()

    filter_str = 'Kittler Filter at SNR {} dB. Threshold at {:2.2f}W'.format(snr, np.mean(t))
    visualise_filter(test_img, mask, filter_str, tracks, 'kittler')

    # out[:, :] = 0
    # out[data >= t] = 255

    return mask, end - start


def rfi_filter(test_img):
    summed = np.sum(test_img, axis=1)
    peaks_snr_i = np.unique(np.where(summed > np.mean(summed) + np.std(summed) * 5.0))

    # Estimate the noise
    noise_mean = np.mean(test_img)

    # Replace the high-peaks channels with the mean noise value
    test_img[peaks_snr_i, :] = noise_mean

    return test_img


# Plot the metrics
# Metrics as a function of SNR
# Filters as a function of speed

if __name__ == '__main__':

    ROOT = "/home/denis/.birales/visualisation/fits"
    OUT_DIR = "/home/denis/.birales/visualisation/analysis"

    OBS_NAME = "NORAD_1328"
    N_TRACKS = 2
    TD = 262144 / 78125 / 32.
    CD = 78125 / 8192.
    F = (1. / TD) / (1. / CD)
    GRADIENT_RANGE = np.array([-0.57, -50.47]) / F
    TRACK_LENGTH_RANGE = np.array([5, 15]) / TD  # in seconds
    TRACK_THICKNESS = 2
    # FITS_FILE = "norad_1328/norad_1328_raw_0.fits"
    FITS_FILE = "filter_test/filter_test_raw_0.fits"
    FITS_FILE = "Observation_2019-05-17T1202/Observation_2019-05-17T1202_raw_1.fits"
    VISUALISE = True
    SAVE_FIGURES = False
    SEED = 56789
    np.random.seed(SEED)

    # snr = [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    # snr = [2, 55]
    snr = [1]
    metrics = {}
    metrics_df = pd.DataFrame()
    filters = [
        # ('Global Filter', global_thres),
        # ('Local Filter', local_thres),
        # ('Otsu Filter', otsu_thres),
        # ('Adaptive Filter', adaptive),
        # ('Kittler', kittler),
        # ('yen', yen),
        # ('iso_data', isodata),
        ('triangle', triangle),
        # ('minimum', minimum),
    ]

    for s in snr:
        metrics[s] = {}

    for s in snr:
        metrics_tmp_df = pd.DataFrame()
        print "\nEvaluating filters with tracks at SNR {:0.2f}W".format(s)
        # Create image from real data
        test_img = create_test_img(os.path.join(ROOT, FITS_FILE), nchans=8192, nsamples=320)

        # Remove channels with RFI
        test_img = rfi_filter(test_img)

        # visualise_image(test_img, 'Test Image: no tracks', tracks=None)

        # Estimate the noise
        noise_mean = np.mean(test_img)

        # Generate a number of tracks
        tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_LENGTH_RANGE, test_img.shape)

        # Add tracks to the true data
        true_image = add_tracks(np.zeros(shape=test_img.shape), tracks, noise_mean, s)

        # Add tracks to the simulated data
        test_img = add_tracks(test_img, tracks, noise_mean, s)

        # visualise_image(test_img, 'Test Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)

        # visualise_image(true_image, 'Truth Image: %d tracks at SNR %dW' % (N_TRACKS, s), tracks)

        # Filter the data in generate metrics
        for f_name, func in filters:
            data = test_img.copy()
            mask, timing = func(data, tracks, s)
            metrics[s][f_name] = evaluate_filter(true_image, data, ~mask, timing, s)

            # mask, timing = hit_and_miss(data, test_img, f_name, mask)
            # metrics[s][f_name + '_hm'] = evaluate_filter(true_image, data, ~mask, timing, s)

            mask, timing = morph_opening(data, test_img, f_name, mask)
            metrics[s][f_name + '_mo'] = evaluate_filter(true_image, data, ~mask, timing, s)

            # feature extraction algorithm

            # dbscan (naive)

            # hough transform

            # edge detectors

            # blob detection

            # astride

            # transform + clustering

        metrics_tmp_df = metrics_tmp_df.from_dict(metrics[s], orient='index')
        metrics_df = metrics_df.append(metrics_tmp_df)

    print metrics_df
