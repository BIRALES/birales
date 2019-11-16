import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
from util import grad, grad2

sns.set(color_codes=True)

# Seaborn settings
# This sets reasonable defaults for font size for
# a figure that will go in a paper
sns.set_context("poster")

# Set the font to be serif, rather than sans
sns.set(font='serif', font_scale=2)

# Make the background white, and specify the
# specific font family
sns.set_style("ticks",
              {'axes.axisbelow': True,
               'axes.edgecolor': '#666666',
               'axes.facecolor': 'white',
               'axes.grid': True,
               'axes.labelcolor': '.15',
               'axes.spines.bottom': True,
               'axes.spines.left': True,
               'axes.spines.right': True,
               'axes.spines.top': True,
               'figure.facecolor': 'white',
               'figure.figsize': (12, 7.416408),
               'font.family': ['serif'],
               'font.sans-serif': ['Arial',
                                   'DejaVu Sans',
                                   'Liberation Sans',
                                   'Bitstream Vera Sans',
                                   'sans-serif'],
               'grid.color': '#e0e0e0',
               'grid.linestyle': '-',
               'image.cmap': 'rocket',
               'lines.solid_capstyle': 'round',
               'lines.linewidth': 5,
               'patch.edgecolor': 'w',
               'patch.force_edgecolor': True,
               'text.color': '.15',
               'xtick.bottom': True,
               'xtick.color': '#666666',
               'xtick.direction': 'out',
               'xtick.top': False,
               'ytick.color': '#666666',
               'ytick.direction': 'out',
               'ytick.left': True,
               'ytick.right': False}
              )

from scipy.stats import linregress

from pybirales.pipeline.modules.detection.msds.util import _partition, __ir2

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


def visualise_ir2(data, org_data, group, ratio2):
    # if len(data) >= 10:
    #     return 0., 0., data

    ms, mc = np.mean(data[:, 1]), np.mean(data[:, 0])
    coords = np.flip(np.swapaxes(data[:, :2] - np.mean(data[:, :2], axis=0), 0, -1), 0)
    eigen_values, eigen_vectors = np.linalg.eig(np.cov(coords))
    sort_indices = np.argsort(eigen_values)[::-1]

    x_v1, y_v1 = eigen_vectors[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = eigen_vectors[:, sort_indices[1]]  # Eigenvector with the second largest eigenvalue

    scale = 6

    ax = plt.axes()
    plt.plot([x_v1 * -scale * 1 + ms, x_v1 * scale * 1 + ms],
             [y_v1 * -scale * 1 + mc, y_v1 * scale * 1 + mc], color='black')
    plt.plot([x_v2 * -scale + ms, x_v2 * scale + ms],
             [y_v2 * -scale + mc, y_v2 * scale + mc], color='blue')

    plt.plot(org_data[:, 1], org_data[:, 0], '.', color='red', markersize=15, zorder=1)
    plt.plot(data[:, 1], data[:, 0], '.', color='blue', markersize=12, zorder=2)
    ratio, _, _ = __ir2(data, min_n=15)

    ax.text(0.05, 0.95,
            'Group: {}\nRatio: {:0.5f}'.format(group, ratio), color='k',
            weight='bold',
            fontsize=15,
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)

    # line_1 = (x1, y1), (x2, y2)
    # line_2 = (x12, y12), (x22, y22)

    # print data[membership(line_1, line_2, data, membership_ratio=0.7), :].shape

    plt.show()


def __plot_leaf(ax, x1, y1, x2, y2, i, score, positive, positives=None, negatives=None):
    from msds import h_cluster
    color = 'r'
    zorder = 1
    lw = 1
    ax.text(x1 + 0.95 * (x2 - x1), y1 + 0.95 * (y2 - y1), i, color='k', fontsize=8, ha='right', va='top', zorder=10)
    if positive:
        color = 'g'
        zorder = 2
        lw = 2

        # Plot mean cluster
        ax.plot(np.mean(positives[:, 1]), np.mean(positives[:, 0]), '*', zorder=10, color='k')

        # Plot the data points that make the cluster
        ax.plot(positives[:, 1], positives[:, 0], 'g.', zorder=3)

        msg = '{:0.2f}\n{:0.2f}'.format(score, grad2(positives))
        ax.text(x1 + 0.5 * (x2 - x1), y1 + 0.5 * (y2 - y1), msg, color='blue', weight='bold',
                fontsize=8, ha='center', va='center')
    else:
        ax.plot(negatives[:, 1], negatives[:, 0], 'r.', zorder=1)
        ax.text(x1 + 0.5 * (x2 - x1), y1 + 0.5 * (y2 - y1), '{}'.format(score), color='k', weight='bold',
                fontsize=8, ha='center', va='center')

        # if i in [1195, 1220]:

        labels, _ = h_cluster(negatives, 2.5, min_length=3, i=i)
        u, c = np.unique(labels, return_counts=True)
        min_mask = c > 3
        u_groups = u[min_mask]

        # start with the smallest grouping
        sorted_groups = u_groups[np.argsort(c[min_mask])]
        for j, g in enumerate(sorted_groups):
            c = negatives[np.where(labels == g)]
            ratio, _, _ = __ir2(c, i=i)
            ax.plot(c[:, 1], c[:, 0], '.', color='pink', zorder=2)

            print 'Cluster: {}, group: {}. ratio:{}. length:{}'.format(i, g, ratio, np.shape(c))

    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=lw, edgecolor=color, facecolor='none',
                             zorder=zorder)

    # Add the patch to the Axes
    ax.add_patch(rect)


def visualise_true_tracks(ax, true_tracks):
    if true_tracks:
        for i, (track_x, track_y) in enumerate(true_tracks):
            ax.plot(track_x, track_y, 'o', color='k', zorder=-1)

    return ax


def set_limits(ax, limits):
    if limits:
        x_start, x_end, y_start, y_end = limits
        ax.set(ylim=(y_start, y_end), xlim=(x_start, x_end))

    return ax


def visualise_clusters(data, cluster_labels, unique_labels, true_tracks, leaves, filename, limits=None, debug=False):
    if debug:
        plt.clf()
        fig, ax = plt.subplots(1)

        for g in unique_labels:
            c = np.vstack(data[cluster_labels == g, 0])
            ax = sns.scatterplot(x=c[:, 1], y=c[:, 0])
            ax.annotate(g, (np.mean(c[:, 1]), np.mean(c[:, 0])))

            # print 'cluster {}: inertia ratio is: {}'.format(g, __ir2(c, 1000, g))

        for i, (cluster, best_gs, msg, x1, x2, y1, y2, n, _, _, _, _, j) in enumerate(leaves):
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='gray', facecolor='none',
                                     zorder=-1)
            ax.text(x1 + 0.95 * (x2 - x1), y1 + 0.95 * (y2 - y1), j, color='k', fontsize=8, ha='right', va='top',
                    zorder=10)
            # Add the patch to the Axes
            ax.add_patch(rect)

        if true_tracks:
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


def visualise_post_processed_tracks(tracks, true_tracks, filename, limits=None, groups=None, debug=False):
    if debug:
        plt.clf()
        fig, ax = plt.subplots(1)
        for i, c in enumerate(tracks):

            # if not np.any(c):
            #     continue
            m, intercept, r_value, p, e = linregress(c[:, 1], c[:, 0])
            group = c[:, 3][0]

            # group = i
            ratio = __ir2(c[:, :2], group)
            x = c[:, 1].mean()
            y = c[:, 0].mean()
            print group, 'R:{:0.5f} P:{:0.5f} E:{:0.5f} I:{:0.5f} N:{} M:{:0.4f} C:{:0.4f} X:{:0.4f} Y:{:0.4f} SNR:{:0.2f}' \
                .format(r_value, p, e, ratio[0], len(c), m, intercept, x, y, np.mean(c[:, 2]))

            ax.annotate('{:0.0f}'.format(group), (x, 1.01 * y), zorder=3)
            if groups:
                if group not in groups:
                    continue

            missing = c[c[:, 3] == -2]
            thickened = c[c[:, 3] == -3]
            detected = c[c[:, 3] >= 0]
            ax = sns.scatterplot(detected[:, 1], detected[:, 0], color='green', marker=".", zorder=4)
            ax = sns.scatterplot(thickened[:, 1], thickened[:, 0], marker=".", color='pink', zorder=2, edgecolor="k")
            ax = sns.scatterplot(missing[:, 1], missing[:, 0], marker="+", color='red', zorder=3, edgecolor="k")

        ax = visualise_true_tracks(ax, true_tracks)

        ax.set(xlabel='Sample', ylabel='Channel')
        ax = set_limits(ax, limits)
        plt.grid()
        ax.figure.savefig(filename)


def visualise_filtered_data(ndx, true_tracks, filename, limits=None, debug=False):
    if debug:
        fig, ax = plt.subplots(1)
        if limits:
            x_start, x_end, y_start, y_end = limits
            ndx = _partition(ndx, x_start, x_end, y_start, y_end)
        ax.plot(ndx[:, 1], ndx[:, 0], '.', 'r', zorder=1)

        ax = visualise_true_tracks(ax, true_tracks)
        ax.set(xlabel='Sample', ylabel='Channel')
        ax = set_limits(ax, limits)
        plt.grid()
        ax.figure.savefig(filename)


def visualise_tree_traversal(ndx, true_tracks, clusters, leaves, filename, limits=None, vis=False):
    if vis:
        plt.clf()
        fig, ax = plt.subplots(1)
        x_start, x_end, y_start, y_end = 0, ndx.shape[0], 0, ndx.shape[1]
        if limits:
            x_start, x_end, y_start, y_end = limits
            ndx = _partition(ndx, x_start, x_end, y_start, y_end)
        ax.plot(ndx[:, 1], ndx[:, 0], 'r.')

        ax = set_limits(ax, limits)

        for i, (cluster, best_gs, msg, x1, x2, y1, y2, n, _, _, _, _, j) in enumerate(leaves):
            cluster_id = i
            if x_start <= x1 <= x_end and y_start <= y1 <= y_end:
                if j != 0:
                    cluster_id = j
                __plot_leaf(ax, x1, y1, x2, y2, cluster_id, msg, False, positives=None, negatives=cluster)

        for i, (cluster, best_gs, msg, x1, x2, y1, y2, n, _, _, _, _, j) in enumerate(clusters):
            cluster_id = i
            if x_start <= x1 <= x_end and y_start <= y1 <= y_end:
                if j != 0:
                    cluster_id = j
                __plot_leaf(ax, x1, y1, x2, y2, cluster_id, msg, True, positives=cluster, negatives=None)

        if true_tracks:
            ax = visualise_true_tracks(ax, true_tracks)
        ax.set(xlabel='Sample', ylabel='Channel')
        ax = set_limits(ax, limits)
        plt.grid()
        ax.figure.savefig(filename)


def plot_metrics(metrics_df):
    # Precision, recall, f1(parameter v, snr)
    # Precision, recall, f1(parameter v, thickness)
    # Precision, recall, f1(parameter v, n candidates)

    algorithms = np.unique(metrics_df.index.values)
    thickness = metrics_df['dx'].unique()
    snr = metrics_df['snr'].unique()

    # F1 vs SNR at different dx
    if len(snr) > 1:
        for t in thickness:
            plt.figure()
            title = "F1 against SNR\n(track width: {:d} px)".format(t)
            data = metrics_df[metrics_df['dx'] == t]
            ax = sns.lineplot(x='snr', y='f1', style=data.index.values, data=data, markers=True, lw=5, ms=10,
                              color='black')
            ax.set_title(title, pad=15)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.set_xlabel("SNR (dB)", labelpad=15)
            ax.set_ylabel("F1 Score", labelpad=15)
            ax.legend().set_title('Detector')
            ax.set(ylim=(0, 1))
            plt.legend(loc='lower right')

    # F1 vs thickness at different SNR
    if len(thickness) > 1:
        for s in snr:
            plt.figure()
            title = "F1 against track width (SNR = {:d} dB)".format(s)
            data = metrics_df[metrics_df['snr'] == s]
            ax = sns.lineplot(x='dx', y='f1', style=data.index.values, data=data, markers=True, lw=5, ms=10,
                              color='black')
            ax.set_title(title)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.set_xlabel("Track width (px)", labelpad=15)
            ax.set_ylabel("F1 Score", labelpad=15)
            ax.legend().set_title('Detector')
            ax.set(ylim=(0, 1))
            plt.legend(loc='lower right')

    # P/R per SNR for each algorithm at dx =1
    # plt.figure()
    # title = "Precision-Recall curve (track width: 1 px)"
    # ax = sns.lineplot(x='recall', y='precision', color='black', style=metrics_df.index.values, markers=True,
    #                   data=metrics_df, lw=5, ms=10)
    # ax.set(ylim=(0, 1), xlim=(0, 1))
    # ax.set_title(title)
    # ax.set_xlabel("Recall")
    # ax.set_ylabel("Precision")
    # # ax.legend().texts[0].set_text('SNR (dB)')

    plt.show()

    print metrics_df[['precision', 'recall', 'snr', 'f1']]


def compare_algorithms(test_image, db_scan_clusters, msds_clusters, iteration, limits=None,
                       debug=False):
    if debug:
        # plt.clf()
        fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
        time_samples = test_image.shape[1]

        ax1 = axes[0]
        ax1.set_title('DBSCAN')
        for i, c in enumerate(db_scan_clusters):
            ax1.plot(c['time_sample'] - time_samples * iteration, c['channel_sample'], 'g.', zorder=4, ms=2)

            print iteration, 'DBSCAN: Cluster {}, N: {}, Unique Samples: {}, Unique Channels: {} Beam:{}'.format(
                i, len(c['time_sample']), len(np.unique(c['time_sample'])), len(np.unique(c['channel_sample'])), c['beam_id'][0])

        ax2 = axes[1]
        ax2.set_title('MSDS')
        for i, c in enumerate(msds_clusters):
            ax2.plot(c['time_sample'] - time_samples * iteration, c['channel_sample'], 'r.', zorder=4, ms=2)

            print 'channel_sample', np.mean(c['channel_sample'])
            print iteration, 'MSDS: Cluster {}, N: {}, Unique Samples: {}, Unique Channels: {}'.format(
                i, len(c['time_sample']), len(np.unique(c['time_sample'])), len(np.unique(c['channel_sample'])))

        ax = axes[2]
        ndx = np.column_stack(np.where(test_image > 0.))
        ndx2 = ndx[np.bitwise_and(ndx[:, 0] > limits[2], ndx[:, 0] < limits[3])]
        ax.plot(ndx2[:, 1], ndx2[:, 0], 'k.', ms=2, zorder=0)
        ax.set_title('Raw Data')
        ax1.set(xlabel='Sample', ylabel='Channel', xticks=range(0, 160, 40))
        ax2.set(xlabel='Sample', xticks=range(0, 160, 40))
        ax.set(xlabel='Sample', xticks=range(0, 160, 40))
        ax = set_limits(ax, limits)
        plt.grid()

        ax.figure.savefig('msds_dbscan_comparison_{}.png'.format(iteration))

        # plt.show()


def visualise_input_data(merged_input, input_data, filtered_merged, beam_id, iteration, debug, limits):
    if not debug:
        return None
    subset = input_data[beam_id, 3618:4618, 0:160]
    # subset = input_data[14,3618:4618, 0:160]
    # subset = np.m(subset, axis=0)
    # subset = np.power(np.abs(subset), 2.0)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Channel")
    im = ax.imshow(subset, aspect='auto', interpolation='none', origin='lower', vmin=0,
                   extent=[0, 160, 3618, 4618])
    fig.colorbar(im)
    fig.tight_layout()
    # plt.title('Input data from {:%H.%M.%S} to {:%M.%S}'.format(time[0], time[1]))
    plt.savefig('input_data_d_{}.png'.format(iteration))
    # plt.show()

    return
    # plt.clf()
    plt.figure(1)
    ch_1 = limits[2]
    ch_n = limits[3]
    ch_1 = 0
    ch_n = 5000
    t1 = 0
    tn = limits[1]
    n_samples = 160

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(10, 10))

    subset = merged_input[ch_1:ch_n, t1:tn]
    im = ax1.imshow(subset, aspect='auto', interpolation='none', origin='lower', vmin=0,
                    extent=[t1, tn, ch_1, ch_n])
    ax1.set(ylabel='Channel', xlabel='Sample', xticks=range(0, n_samples, 50), title='Merged')

    subset = input_data[beam_id, ch_1:ch_n, t1:tn]

    subset = np.power(np.abs(subset), 2.0)

    im1 = ax2.imshow(subset, aspect='auto', interpolation='none', origin='lower', vmin=0,
                     extent=[t1, tn, ch_1, ch_n])
    ax2.set(xlabel='Sample', xticks=range(0, n_samples, 50), title='Beam %s' % beam_id)

    subset = filtered_merged[ch_1:ch_n, :n_samples]
    im = ax3.imshow(subset, aspect='auto', interpolation='none', origin='lower', vmin=0,
                    extent=[t1, tn, ch_1, ch_n])
    ax3.set(xlabel='Sample', xticks=range(0, n_samples, 50), title='Merged & Filtered')

    fig.colorbar(im1)
    ax1.figure.savefig('input_data_{}.png'.format(iteration))

    # plt.show()
    # exit()
