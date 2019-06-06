import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

plt.rcParams['figure.figsize'] = (12, 10)


def save_figure(filename, out_dir, obs_name, save=False):
    if save:
        plt.savefig(os.path.join(out_dir, obs_name, filename, '.pdf'))


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


def visualise_image(image, title, tracks, visualise=False):
    """
    Visualise the test data
    :param image:
    :return:
    """
    if visualise:
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


def visualise_filter(data, mask, tracks, f_name, snr, threshold=None, visualise=False):
    if visualise:
        filter_str = '{} at SNR {} dB.'.format(f_name, snr)
        if threshold:
            filter_str += 'Threshold at {:2.2f}W'.format(threshold)

        x_start, x_end, y_start, y_end = get_limits(data, tracks)
        ax = sns.heatmap(data[y_start:y_end, x_start:x_end], cbar_kws={'label': 'Power (dB)'},
                         xticklabels=25,
                         yticklabels=25,
                         mask=mask[y_start:y_end, x_start:x_end])
        # ax.set(ylim=(y_start, y_end), xlim=(x_start, x_end))
        ax.invert_yaxis()
        ax.set(xlabel='Time sample', ylabel='Channel', title=filter_str)
        plt.show()

        print 'Showing: ' + filter_str


def visualise_detector(data, candidates, tracks, d_name, snr, visualise=False):
    if visualise:
        title = '{} at SNR {} dB.'.format(d_name, snr)
        x_start, x_end, y_start, y_end = get_limits(data, tracks)

        ax = plt.axes()
        for t in tracks:
            x, y = t
            ax.scatter(x, y, c='black', marker='s')


        for c in candidates:
            channel = c[:, 0].astype(int)
            time = c[:, 1].astype(int)

            if np.any(channel) and np.any(time):
                ax.scatter(time, channel)

        ax.invert_yaxis()
        ax.set(ylim=(y_start, y_end), xlim=(x_start, x_end))
        ax.set(xlabel='Time sample', ylabel='Channel', title=title)
        ax.legend()
        plt.show()

        print 'Showing: ' + title
