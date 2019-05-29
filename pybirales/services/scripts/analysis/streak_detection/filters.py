import numpy as np
from scipy.ndimage import binary_hit_or_miss, binary_opening
from skimage.filters import *


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_moving_average(chunked_data, iter, n=3):
    n_mean = np.array([np.mean(c) for c in chunked_data[:iter+1]])
    n_std = np.array([np.std(c) for c in chunked_data[:iter+1]])

    s = len(n_mean) - n

    if s < 0:
        s = 0

    return np.mean(n_mean[s:]), np.std(n_std[s:])


def get_moving_average_local(chunked_data, iter, n=3):
    n_mean = np.array([np.mean(c, axis=1) for c in chunked_data[:iter+1]])
    n_std = np.array([np.std(c, axis=1) for c in chunked_data[:iter+1]])

    s = len(n_mean) - n

    if s < 0:
        s = 0

    return np.mean(n_mean[s:], axis=0), np.std(n_std[s:], axis=0)


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


def global_thres(test_img):
    channel_noise = np.mean(test_img)
    std = np.std(test_img)

    threshold = 4 * std + channel_noise
    global_threshold_mask = test_img < threshold

    return global_threshold_mask, threshold


def local_thres(test_img):
    channel_noise = np.mean(test_img, axis=1)
    std = np.std(test_img, axis=1)

    threshold = 4 * std + channel_noise

    # print 'Local', np.max(threshold), np.min(threshold), np.std(threshold)
    local_threshold_mask = test_img < np.expand_dims(threshold, axis=1)

    return local_threshold_mask, np.mean(threshold)


def global_thres_running(test_img, channel_noise, std):
    threshold = 5 * std + channel_noise
    global_threshold_mask = test_img < threshold

    return global_threshold_mask, threshold


def local_thres_running(test_img, channel_noise, std):
    threshold = 5 * std + channel_noise

    local_threshold_mask = test_img < np.expand_dims(threshold, axis=1)

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
