import numpy as np
from scipy.ndimage import binary_hit_or_miss, binary_opening
from skimage.filters import *
from skimage.feature import canny
from scipy.signal import find_peaks_cwt

def get_moving_average(chunked_data, iter, n=5):
    n_mean = np.array([np.mean(c) for c in chunked_data[:iter+1]])
    n_std = np.array([np.std(c) for c in chunked_data[:iter+1]])

    s = len(n_mean) - n

    if s < 0:
        s = 0

    return np.mean(n_mean[s:]), np.mean(n_std[s:])


def get_moving_average_local(chunked_data, iter, n=5):
    n_mean = np.array([np.mean(c, axis=1) for c in chunked_data[:iter+1]])
    n_std = np.array([np.std(c, axis=1) for c in chunked_data[:iter+1]])

    s = len(n_mean) - n

    if s < 0:
        s = 0

    return np.mean(n_mean[s:], axis=0), np.mean(n_std[s:], axis=0)


def chunked_filtering(test_img, filter_func):
    # chunk the data
    N_chunks = test_img.shape[1] / 32
    c_test_img = np.array_split(test_img, N_chunks, axis=1)

    result_mask = np.full(np.shape(test_img), False)

    # filter it in chunks using the chosen filter
    threshold = 0.
    for i, c in enumerate(c_test_img):
        if filter_func is global_thres_running:
            channel_noise, std = get_moving_average(c_test_img, i)
            mask, threshold = filter_func(np.array(c), channel_noise, std)

        elif filter_func is local_thres_running:
            channel_noise, std = get_moving_average_local(c_test_img, i)
            # print channel_noise, std
            mask, threshold = filter_func(np.array(c), channel_noise, std)

        else:
            mask, threshold = filter_func(np.array(c))

        result_mask[:, i * 32: i * 32 + 32] = mask

    return result_mask, threshold

def no_filter(test_img):
    return np.zeros(test_img.shape, dtype=bool), None

def global_thres(test_img):
    channel_noise = np.mean(test_img)
    std = np.std(test_img)

    threshold = 4 * std + channel_noise
    global_threshold_mask = test_img < threshold

    # print 'Global Noise', np.mean(channel_noise), np.mean(std)

    return global_threshold_mask, threshold

def global_thres_running(test_img, channel_noise, std):
    threshold = 4 * std + channel_noise
    global_threshold_mask = test_img < threshold

    # print 'Global Noise (R)', np.mean(channel_noise), np.mean(std)

    return global_threshold_mask, threshold

def local_thres(test_img):
    channel_noise = np.mean(test_img, axis=1)
    std = np.std(test_img, axis=1)

    threshold = 4 * std + channel_noise

    # print 'Local Noise', np.mean(channel_noise), np.mean(threshold)

    local_threshold_mask = test_img < np.expand_dims(threshold, axis=1)

    return local_threshold_mask, np.mean(threshold)

def local_thres_running(test_img, channel_noise, std):
    threshold = 4 * std + channel_noise

    local_threshold_mask = test_img < np.expand_dims(threshold, axis=1)

    # print 'Local Noise (R)', np.mean(channel_noise), np.mean(std)

    # print 'Running', np.mean(threshold), np.shape(threshold), np.mean(channel_noise), np.mean(std)
    return local_threshold_mask, np.mean(threshold)


def otsu_thres(test_img):
    """
    Otsu Filter
    :param test_img:
    :param true_img:
    :param snr:
    :return:
    """

    global_thresh = threshold_otsu(test_img)
    global_filter_mask = test_img < global_thresh

    return global_filter_mask, global_thresh

def canny_filter(test_img):
    return canny(test_img, 2), None


def adaptive(test_img):
    """
    Adaptive Threshold filter
    :param test_img:
    :param true_img:
    :param snr:
    :return:
    """

    block_size = 151
    local_thresh = threshold_local(test_img, block_size=block_size)
    local_filter_mask = test_img < local_thresh

    return local_filter_mask, local_thresh


def yen(test_img):
    local_thresh = threshold_yen(test_img, nbins=1024)
    local_filter_mask = test_img < local_thresh

    return local_filter_mask, local_thresh


def minimum(test_img):
    local_thresh = threshold_minimum(test_img, nbins=1024)
    local_filter_mask = test_img < local_thresh
    return local_filter_mask, local_thresh


def triangle(test_img):
    local_thresh = threshold_triangle(test_img, nbins=1024)
    local_filter_mask = test_img < local_thresh

    return local_filter_mask, local_thresh


def isodata(test_img):
    local_thresh = threshold_isodata(test_img, nbins=1024)
    local_filter_mask = test_img < local_thresh

    return local_filter_mask, local_thresh


def hit_and_miss(data, test_img, mask):
    structure = np.zeros((5, 5))
    structure[2, 2] = 1

    mask += binary_hit_or_miss(data, structure1=structure)

    data[~mask] = test_img[~mask]

    return mask, None


def morph_opening(data, test_img, mask):
    tmp = binary_opening(data, structure=np.ones((2, 2))).astype(np.int)
    mask += tmp < 1

    data[~mask] = test_img[~mask]

    return mask, None


def kittler(test_img):
    h, g = np.histogram(test_img.ravel(), bins=1024)
    h = h.astype(np.float)
    g = g.astype(np.float)

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

    mask = test_img <= t

    return mask, np.mean(t)


def rfi_filter(test_img):
    summed = np.sum(test_img, axis=1)
    peaks_snr_i = np.unique(np.where(summed > np.mean(summed) + np.std(summed) * 5.0))

    # Estimate the noise
    noise_mean = np.mean(test_img)

    # Replace the high-peaks channels with the mean noise value
    test_img[peaks_snr_i, :] = noise_mean

    return test_img


def cfar(test_image):
    """
        Detect peaks with CFAR algorithm.

        num_train: Number of training cells.
        num_guard: Number of guard cells.
        rate_fa: False alarm rate.
        """


    def cfar_1d(x):
        # x = np.ones(shape=(100,))
        # x[10:20] = 5
        num_train = 10
        num_guard = 5
        rate_fa = 0.9
        num_cells = x.size

        num_train_half = int(round(num_train / 2))
        num_guard_half = int(round(num_guard / 2))
        num_side = int(num_train_half + num_guard_half)

        alpha = num_train * (rate_fa ** (-1. / num_train) - 1)  # threshold factor
        peak_idx = []

        for i in range(num_side, num_cells - num_side):

            if i != i - num_side + np.argmax(x[i - num_side:i + num_side + 1]):
                continue

            sum1 = np.sum(x[i - num_side:i + num_side + 1])
            sum2 = np.sum(x[i - num_guard_half:i + num_guard_half + 1])
            p_noise = (sum1 - sum2) / num_train
            threshold = alpha * p_noise
            # print threshold, alpha, p_noise, sum1, sum2, num_train
            if x[i] > threshold:
                peak_idx.append(i)

            # print threshold

        peak_idx = np.array(peak_idx, dtype=int)
        # print peak_idx
        y = np.ones(shape=num_cells, dtype=bool)
        y[~peak_idx] = False

        # print y

        # print peak_idx, threshold
        peak_idx = find_peaks_cwt(x, np.arange(1,5) )

        # peak_idx = np.array(peak_idx, dtype=int)
        # print peak_idx
        y = np.ones(shape=num_cells, dtype=bool)
        y[~peak_idx] = False

        return y

    # print 'Test image', np.mean(test_image)
    mask = np.apply_along_axis(cfar_1d, 1, test_image)
    # print 'Mask', np.mean(mask)
    return mask, None