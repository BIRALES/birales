"""
This script evaluates the performance of different filters for
image segmentation of the spectrogram produced by the PyBirales
Channeliser.

Notebook reproduces the figures used in the
streak detection paper.

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from skimage.filters import threshold_otsu, threshold_local
from sklearn.metrics import jaccard_similarity_score

plt.rcParams['figure.figsize'] = (12, 10)


def save_figure(filename):
    if SAVE_FIGURES:
        plt.savefig(os.path.join(OUT_DIR, OBS_NAME, filename, '.pdf'))


def calculate_amplitude(noise_std, snr):
    """

    :param noise_std:
    :param snr:
    :return:
    """
    noise_avg_watts = noise_std ** 2
    noise_avg_db = 10 * np.log10(noise_avg_watts)

    signal_avg_db = noise_avg_db + snr
    sig_avg_watts = 10 ** (signal_avg_db / 10)

    # we can use this as amplitude for our chirp
    return np.sqrt(sig_avg_watts)


def generate_noise(mean_noise_power, std, n_samples):
    return np.random.normal(loc=mean_noise_power, scale=std, size=n_samples)


def visualise_image(image):
    # Visualise the test data
    ax = sns.heatmap(image, cbar_kws={'label': 'Power (dB)'},
                     xticklabels=25,
                     yticklabels=25)
    ax.invert_yaxis()
    ax.set(xlabel='Time sample', ylabel='Channel')

    plt.show()


def uniform(min_max_range):
    return np.random.uniform(min_max_range[0], min_max_range[1])


def create_track(x, gradient, intercept, img_shape):
    y = x * gradient + intercept
    y[y <= 0] = 0
    y[y >= img_shape[0]] = img_shape[0] - 1
    return y.astype(int)


def create_test_image(filepath):
    fits_file = fits.open(filepath)
    test_image = fits_file[0].data[0][8000:, 1400:]
    print 'Test Image of size {} was generated'.format(test_image.shape)

    return test_image


def jaccard(true_image, filtered_image):
    true_image = np.array(true_image).ravel()
    filtered_image = np.array(filtered_image).ravel()
    return jaccard_similarity_score(true_image, filtered_image)


def get_test_tracks(n_tracks, gradient, track_start, image_shape, noise_mean, snr):
    tracks = []
    for i in range(0, n_tracks):
        m = uniform(gradient)
        start = np.random.randint(low=track_start[0], high=track_start[1])
        end = np.random.randint(low=start, high=image_shape[1])
        x = np.arange(start, end)
        c = np.random.randint(low=100, high=image_shape[0])
        y = create_track(x, m, c, image_shape)

        print 'Created track with m={:0.2f}, c={:0.2f}, of {} pixels at x={} with SNR={:0.2f} dB'.format(m, c, (
                max(x) - start), start, SNR)

        #         image[y, x] += calculate_amplitude(noise_std=NOISE_MEAN, snr=SNR)

        tracks.append((x, y, calculate_amplitude(noise_std=noise_mean, snr=snr)))
    return tracks


def add_tracks(image, tracks):
    for track in tracks:
        image[track[1], track[0]] += track[2]

    return image


def threshold_nchans(data):
    """
    Background noise filter: Show noise distribution (histogram) before and after filter is applied

    :param data:
    :return:
    """
    channel_noise = np.mean(data, axis=1)
    std = np.std(data, axis=1)

    threshold = 2 * std + channel_noise

    return threshold


def threshold_global(data):
    channel_noise = np.mean(data)
    std = np.std(data)

    threshold = 2 * std + channel_noise

    return threshold


def visualise_filter(data, mask, threshold, filename=None):
    ax = sns.heatmap(data, cbar_kws={'label': 'Power (dB)'},
                     xticklabels=25,
                     yticklabels=25,
                     mask=mask
                     )
    ax.invert_yaxis()
    ax.set(xlabel='Time sample', ylabel='Channel')

    print('Noise: {:0.2f} dB, Thres: {:0.2f} dB (~SNR: {:0.2f} dB)'.format(np.mean(data), np.mean(threshold),
                                                                           np.mean(threshold / np.mean(data))))
    plt.show()

def evaluate_filter(truth, prediction):
    """

    :param truth:
    :param prediction:
    :return:
    """
    # jaccard = jaccard(truth, prediction)
    # si = similarity_index(truth, prediction)
    # recall
    # precision
    # accuracy
    return

np.random.seed(56789)

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
TRACK_START = np.array([0, 10]) / TD  # in seconds
FITS_FILE = "norad_1328/norad_1328_raw_0.fits"

# Create image from real data
TEST_IMAGE = create_test_image(os.path.join(ROOT, FITS_FILE))

# Estimate the noise
NOISE_MEAN = np.mean(TEST_IMAGE)
NOISE_STD = np.std(TEST_IMAGE)

# Generate a number of tracks
tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_START, TEST_IMAGE.shape, NOISE_MEAN, SNR)

TRUE_IMAGE = add_tracks(np.zeros(shape=TEST_IMAGE.shape), tracks)
TEST_IMAGE = add_tracks(TEST_IMAGE, tracks)

def global_thres(test_img, true_img):
    # BF Global Filter
    threshold = threshold_global(test_img)
    global_threshold_mask = np.where(test_img < threshold, threshold, 0)
    visualise_filter(test_img, global_threshold_mask, threshold, 'global_filter')
    return evaluate_filter(true_img, global_threshold_mask)

def local_thres(test_img, true_img):
    # BF Local Filter
    threshold = threshold_nchans(test_img)
    local_threshold_mask = np.where(test_img < np.expand_dims(threshold, axis=1), threshold, 0)
    visualise_filter(test_img, local_threshold_mask, threshold, 'local_filter')
    return evaluate_filter(true_img, TEST_IMAGE[local_threshold_mask])

def otsu_thres(test_img, true_img):
    # Otsu Filter
    global_thresh = threshold_otsu(test_img)
    global_filter_mask = np.where(test_img < global_thresh, TEST_IMAGE, 0)
    visualise_filter(test_img, global_filter_mask, global_thresh, 'after_global_otsu_filter')
    return evaluate_filter(true_img, TEST_IMAGE[global_filter_mask])

def adaptive(test_img, true_img):
    # Adaptive Threshold filter
    block_size = 5
    local_thresh = threshold_local(test_img, block_size, offset=2)
    local_filter_mask = np.where(test_img < local_thresh, TEST_IMAGE, 0)
    visualise_filter(test_img, local_filter_mask, local_thresh, 'after_local_filter')
    return evaluate_filter(true_img, TEST_IMAGE[local_filter_mask])

def median(test_img, true_img):
    pass


snr = [0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55]
metrics = {}
for s in snr:
    metrics[s] = []

    # Create image from real data
    test_image = create_test_image(os.path.join(ROOT, FITS_FILE))

    # Estimate the noise
    NOISE_MEAN = np.mean(test_image)

    # Generate a number of tracks
    tracks = get_test_tracks(N_TRACKS, GRADIENT_RANGE, TRACK_START, test_image.shape, NOISE_MEAN, s)

    true_image = add_tracks(np.zeros(shape=test_image.shape), tracks)
    test_image = add_tracks(test_image, tracks)

    metrics[s] += global_thres(test_image, true_image)

    metrics[s] += local_thres(test_image, true_image)

    metrics[s] += otsu_thres(test_image, true_image)

    metrics[s] += adaptive(test_image, true_image)

    metrics[s] += median(test_image, true_image)

# Plot the metrics
# Metrics as a function of SNR
# Filters as a function of speed