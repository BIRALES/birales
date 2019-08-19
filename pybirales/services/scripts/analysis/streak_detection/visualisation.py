import os
from itertools import cycle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress

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

        ax.set(xlabel='Time sample', ylabel='Channel')
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


def __plot_leave(ax, x1, y1, x2, y2, i, score, positive, positives=None, negatives=None):
    color = 'r'
    zorder = 1
    lw = 1
    if positive:
        color = 'g'
        zorder = 2
        lw = 2
    if np.any(positives):
        ax.plot(positives[:, 1], positives[:, 0], 'g.', zorder=3)
    colors = cycle(['b', 'm', 'c', 'y', 'g', 'orange', 'indianred', 'aqua', 'darkcyan', 'mediumpurple'])

    if negatives > 0:
        if not positive:
            ax.plot(positives[:, 1], positives[:, 0], 'r.', zorder=4)

        # scores = '{}\n'.format(i)
        scores = ''
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if len(negatives) > 50:
            print i, len(negatives), x1, y1, x2, y2
        for j, (n, ratio) in enumerate(negatives):
            ax.plot(n[:, 1], n[:, 0], '.', zorder=5, color=next(colors))
            scores += '{}: {:0.3f}\n'.format(j, ratio)
        ax.text(x1 * 1.01, y1 + 0.95 * (y2 - y1), scores, color='k', fontsize=10, va='top')

    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=lw, edgecolor=color, facecolor='none',
                             zorder=zorder)

    # Add the patch to the Axes
    ax.add_patch(rect)

    if positive:
        ax.text(x1 + 0.5 * (x2 - x1), y1 + 0.5 * (y2 - y1), score, color='k', weight='bold',
                fontsize=10, ha='center', va='center')

    ax.text(x1 + 0.95 * (x2 - x1), y1 + 0.95 * (y2 - y1), i, color='k', fontsize=8, ha='right', va='top', zorder=10)


def visualise_true_tracks(ax, true_tracks):
    for i, (track_x, track_y) in enumerate(true_tracks):
        ax.plot(track_x, track_y, 'o', color='k', zorder=-1)

    return ax


def set_limits(ax, limits):
    if limits:
        x_start, x_end, y_start, y_end = limits
        ax.set(ylim=(y_start, y_end), xlim=(x_start, x_end))

    return ax


def visualise_clusters(data, cluster_labels, unique_labels, true_tracks, filename, limits=None, debug=False):
    if debug:
        plt.clf()
        fig, ax = plt.subplots(1)
        for g in unique_labels:
            c = np.vstack(data[cluster_labels == g, 0])
            ax = sns.scatterplot(x=c[:, 1], y=c[:, 0])
            ax.annotate(g, (np.mean(c[:, 1]), np.mean(c[:, 0])))

        ax = visualise_true_tracks(ax, true_tracks)

        ax.set(xlabel='Sample', ylabel='Channel', title='Detected Clusters')
        ax = set_limits(ax, limits)
        plt.grid()
        ax.figure.savefig(filename)


def visualise_candidates(candidates, true_tracks, filename, limits=None, debug=False):
    if debug:
        plt.clf()
        fig, ax = plt.subplots(1)
        for c in candidates:
            group_2 = c[:, 3][0]
            ax = sns.scatterplot(x=c[:, 1], y=c[:, 0])
            ax.annotate(group_2, (np.mean(c[:, 1]), np.mean(c[:, 0])))

        ax = visualise_true_tracks(ax, true_tracks)

        ax.set(xlabel='Sample', ylabel='Channel', title='Detected Candidates')
        ax = set_limits(ax, limits)
        plt.grid()
        ax.figure.savefig(filename)


def visualise_tracks(tracks, true_tracks, filename, limits=None, debug=False):
    if debug:
        plt.clf()
        fig, ax = plt.subplots(1)
        for c in tracks:
            group_2 = c[:, 3][0]
            ax = sns.scatterplot(x=c[:, 1], y=c[:, 0])
            ax.annotate(group_2, (np.mean(c[:, 1]), np.mean(c[:, 0])))

        ax = visualise_true_tracks(ax, true_tracks)

        ax.set(xlabel='Sample', ylabel='Channel', title='Detected Tracks')
        ax = set_limits(ax, limits)
        plt.grid()
        ax.figure.savefig(filename)


def visualise_post_processed_tracks(tracks, true_tracks, filename, limits=None, debug=False):
    if debug:
        plt.clf()
        fig, ax = plt.subplots(1)
        for i, c in enumerate(tracks):
            m, intercept, r_value, p, e = linregress(c[:, 1], c[:, 0])
            group = c[c[:, 3] > 0][:, 3][0]
            ratio = __ir(c[:, :2], group)
            x = c[:, 1].mean()
            y = c[:, 0].mean()

            print group, 'R:{:0.5f} P:{:0.5f} E:{:0.5f} I:{:0.5f} N:{} M:{:0.4f} C:{:0.4f} X:{:0.4f} Y:{:0.4f}' \
                .format(r_value, p, e, ratio, len(c), m, intercept, x, y)

            missing = c[c[:, 3] == -2]
            thickened = c[c[:, 3] == -3]
            detected = c[c[:, 3] >= 0]
            ax = sns.scatterplot(detected[:, 1], detected[:, 0], color='green', marker=".", zorder=4)
            ax = sns.scatterplot(thickened[:, 1], thickened[:, 0], marker=".", color='pink', zorder=2, edgecolor="k")
            ax = sns.scatterplot(missing[:, 1], missing[:, 0], marker="+", color='red', zorder=3, edgecolor="k")

            ax.annotate('Group {:0.0f}'.format(group), (x, 1.01 * y), zorder=3)

            if int(group) == 77:
                for b in c:
                    print b[0], b[1], b[2], b[3]

        ax = visualise_true_tracks(ax, true_tracks)

        ax.set(xlabel='Sample', ylabel='Channel')
        ax = set_limits(ax, limits)
        plt.grid()
        ax.figure.savefig(filename)


def visualise_filtered_data(ndx, true_tracks, filename, limits=None, debug=False, vis=False):
    if vis or debug:
        fig, ax = plt.subplots(1)
        if limits:
            x_start, x_end, y_start, y_end = limits
            ndx = _partition(ndx, x_start, x_end, y_start, y_end)
        ax.plot(ndx[:, 1], ndx[:, 0], '.', 'r', zorder=-3)

        ax = visualise_true_tracks(ax, true_tracks)
        ax.set(xlabel='Sample', ylabel='Channel')
        ax = set_limits(ax, limits)
        plt.grid()
        ax.figure.savefig(filename)


def visualise_tree_traversal(ndx, true_tracks, leaves, rectangles, filename, limits=None, vis=False):
    if vis:
        fig, ax = plt.subplots(1)
        x_start, x_end, y_start, y_end = 0, ndx.shape[0], 0, ndx.shape[1]
        if limits:
            x_start, x_end, y_start, y_end = limits
            ndx = _partition(ndx, x_start, x_end, y_start, y_end)
        ax.plot(ndx[:, 1], ndx[:, 0], '.', 'r')
        ax = set_limits(ax, limits)

        for i, (cluster, rejected, best_gs, msg, x1, x2, y1, y2, n) in enumerate(rectangles):
            if x_start <= x1 <= x_end and y_start <= y1 <= y_end:
                __plot_leave(ax, x1, y1, x2, y2, i, msg, False, positives=cluster, negatives=rejected)

        for i, (cluster, best_gs, msg, x1, x2, y1, y2, n, _, _, _) in enumerate(leaves):
            if x_start <= x1 <= x_end and y_start <= y1 <= y_end:
                __plot_leave(ax, x1, y1, x2, y2, i, msg, True, positives=cluster, negatives=None)

        ax = visualise_true_tracks(ax, true_tracks)
        ax.set(xlabel='Sample', ylabel='Channel')
        ax = set_limits(ax, limits)
        plt.grid()
        ax.figure.savefig(filename)
