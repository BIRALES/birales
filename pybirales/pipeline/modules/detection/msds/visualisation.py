import datetime
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from cycler import cycler
from matplotlib.font_manager import FontProperties

from pybirales.pipeline.modules.detection.msds.msds import h_cluster
from pybirales.pipeline.modules.detection.msds.util import grad2, __ir, missing_score
from pybirales.services.scripts.analysis.analysis import COLORS

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
               'axes.edgecolor': '#333333',
               'axes.facecolor': 'white',
               'axes.grid': True,
               'axes.labelcolor': '.15',
               'axes.spines.bottom': True,
               'axes.spines.left': True,
               'axes.spines.right': True,
               'axes.spines.top': True,
               'axes.linewidth': 1.25,
               'figure.facecolor': 'white',
               'figure.figsize': (12, 7.416408),
               'font.family': ['serif'],
               'font.sans-serif': ['Arial',
                                   'DejaVu Sans',
                                   'Liberation Sans',
                                   'Bitstream Vera Sans',
                                   'sans-serif'],
               'grid.color': '#e0e0e0',
               'grid.linestyle': '--',
               'image.cmap': 'rocket',
               'lines.solid_capstyle': 'round',
               'lines.linewidth': 5,
               'patch.edgecolor': 'b',
               'patch.force_edgecolor': True,
               'text.color': '.15',
               'xtick.bottom': True,
               'xtick.color': '#333333',
               'xtick.direction': 'out',
               'xtick.top': False,
               'ytick.color': '#333333',
               'ytick.direction': 'out',
               'ytick.left': True,
               'ytick.right': False}
              )

from scipy.stats import linregress

from pybirales.pipeline.modules.detection.msds.util import _partition, __ir2

plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['axes.edgecolor'] = "#333333"
plt.rc('ytick', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=15)


def save_figure(filename, out_dir, obs_name, save=False):
    if save:
        plt.savefig(os.path.join(out_dir, obs_name, filename, '.pdf'))


def round_down(x):
    return int(np.floor(x / 10.0)) * 10


def round_up(x):
    return int(np.ceil(x / 10.0)) * 10


def get_limits(image, tracks):
    x_start = 0
    x_end = image.shape[1]

    y_start = 0
    y_end = image.shape[0]

    # return x_start, x_end, y_start, y_end
    if tracks:
        m_track = tracks[0]
        for t in tracks:
            if len(t[0]) < len(m_track[0]):
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
    return round_down(x_start), round_up(x_end), round_down(y_start), round_up(y_end)


def visualise_image(image, title, tracks, visualise=False, file_name=None, bar=True):
    """
    Visualise the test data
    :param image:
    :return:
    """
    if visualise:
        x_start, x_end, y_start, y_end = get_limits(image, tracks)
        prev_shape = image.shape

        print("Visualising a subset of the image {} from {}".format(image[y_start:y_end, x_start:x_end].shape,
                                                                    prev_shape))

        ax = sns.heatmap(image[y_start:y_end, x_start:x_end], xticklabels=20, yticklabels=50, cbar=True)
        ax.invert_yaxis()

        ax.set(xlabel='Time sample', ylabel='Channel')
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.tick_params(width=2)

        ax.set(xlabel='Time sample', ylabel='Channel')
        for _, spine in ax.spines.items():
            spine.set_visible(True)

        if file_name:
            plt.tight_layout()
            plt.savefig(file_name)

        plt.show()


def visualise_filter(data, mask, tracks, f_name, snr, threshold=None, visualise=True, file_name=None):
    if visualise:
        filter_str = '{} at SNR {} dB.'.format(f_name, snr)
        if threshold:
            filter_str += 'Threshold at {:2.2f}W'.format(threshold)

        x_start, x_end, y_start, y_end = get_limits(data, tracks)
        ax = sns.heatmap(data[y_start:y_end, x_start:x_end],
                         xticklabels=20,
                         yticklabels=50,
                         mask=mask[y_start:y_end, x_start:x_end])
        # ax.set(ylim=(y_start, y_end), xlim=(x_start, x_end))
        ax.invert_yaxis()
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.tick_params(width=2)
        # ax.set(xlabel='Time sample', ylabel='Channel', title=filter_str)
        ax.set(xlabel='Time sample', ylabel='Channel')
        for _, spine in ax.spines.items():
            spine.set_visible(True)

        print('Showing: ' + filter_str)

        if file_name:
            plt.tight_layout()
            plt.savefig(file_name)

        plt.show()


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


def visualise_ir2_pub(data, org_data, group, ratio2):
    # if len(data) >= 10:
    #     return 0., 0., data

    ms, mc = np.mean(data[:, 1]), np.mean(data[:, 0])
    coords = np.flip(np.swapaxes(data[:, :2] - np.mean(data[:, :2], axis=0), 0, -1), 0)
    eigen_values, eigen_vectors = np.linalg.eig(np.cov(coords))
    sort_indices = np.argsort(eigen_values)[::-1]

    x_v1, y_v1 = eigen_vectors[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = eigen_vectors[:, sort_indices[1]]  # Eigenvector with the second largest eigenvalue

    scale_1 = 25
    scale_2 = 2

    # ax = plt.axes()
    fig, ax = plt.subplots(1)
    ax.plot([x_v1 * -scale_1 * 1 + ms, x_v1 * scale_1 * 1 + ms],
            [y_v1 * -scale_1 * 1 + mc, y_v1 * scale_1 * 1 + mc], color='black')
    ax.plot([x_v2 * -scale_2 + ms, x_v2 * scale_2 + ms],
            [y_v2 * -scale_2 + mc, y_v2 * scale_2 + mc], color='gray')

    # plt.plot(org_data[:, 1], org_data[:, 0], '.', color='red', markersize=15, zorder=1)
    ax.plot(data[:, 1], data[:, 0], '.', color='k', markersize=12, zorder=2)
    ratio, _, _ = __ir2(data, min_n=15)

    # ax.text(0.05, 0.95,
    #         'Ratio: {:0.3f}'.format(ratio), color='k',
    #         weight='bold',
    #         fontsize=15,
    #         horizontalalignment='right',
    #         verticalalignment='center', transform=ax.transAxes)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.set(xlabel='Sample', ylabel='Channel')
    plt.tight_layout()
    plt.grid()
    ax.figure.savefig('5_g={}_ir={:2.3}.pdf'.format(group, ratio))


def __plot_leaf(ax, x1, y1, x2, y2, i, score, positive, positives=None, negatives=None, pub=False):
    color = 'r'
    zorder = 1
    lw = 2
    figure_limits = (70, 4010)
    # if not False:
    if not positive:
        ax.text(x1 + 0.95 * (x2 - x1), y1 + 0.95 * (y2 - y1), i, color='k', fontsize=8, ha='right', va='top', zorder=10)

    # if i == 343 and positive:
    #     visualise_ir2_pub(positives, positives, i, -1)
    #     m = positives.shape[0]
    #     empty_dm = np.full((m * (m - 1)) // 2, 10000, dtype=np.float)
    #     distances = dm(positives, m, empty_dm, -3.51929755, i)
    #
    #     # Perform single linkage clustering
    #     Z = linkage(distances, 'single')
    #
    #     # Determine to which cluster each initial point would belong given a distance threshold
    #     labels = fcluster(Z, 4, criterion='distance')
    #     viz_den(positives, Z, 4, labels)

    if positive:
        color = 'g'
        zorder = 2
        lw = 2

        # Plot mean cluster
        if not pub:
            ax.plot(np.mean(positives[:, 1]), np.mean(positives[:, 0]), '*', zorder=10, color='w', markeredgecolor='k',
                    markersize=7.5)

        # Plot the data points that make the cluster
        ax.plot(positives[:, 1], positives[:, 0], 'g.', zorder=3)

        if not pub:
            msg = '{:0.4f}\n{:0.3f}'.format(score, grad2(positives))
            ax.text(x1 + 0.5 * (x2 - x1), y1 + 0.5 * (y2 - y1), msg, color='blue', weight='bold',
                    fontsize=8, ha='center', va='center')
    else:
        # ax.plot(negatives[:, 1], negatives[:, 0], '.', color='gray', zorder=1)

        if not pub:
            ax.text(x1 + 0.5 * (x2 - x1), y1 + 0.5 * (y2 - y1), '{}'.format(score), color='k', weight='bold',
                    fontsize=8, ha='center', va='center')
            import random
            labels, u_groups = h_cluster(negatives, 4, min_length=5, i=i)
            colors = ['pink', 'b', 'k', 'c', 'm', 'y']
            for j, g in enumerate(u_groups):
                c = negatives[np.where(labels == g)]

                ms = missing_score(negatives[:, 1])
                ratio = __ir(c, min_n=10, i=g)

                ax.plot(c[:, 1], c[:, 0], '.', color=random.choice(colors), zorder=20)

                print('Leave: {}, group: {}. ratio:{:0.3f}, ms: {:0.3f} length:{}:'.format(i, g, ratio,
                                                                                           ms,
                                                                                           np.shape(c)))

            # colors = ['pink', 'b', 'k', 'c', 'm', 'y']
            # for j, g in enumerate(sorted_groups):
            #     c = negatives[np.where(labels == g)]
            #     ratio, _, _ = __ir2(c, i=i)
            #     ax.plot(c[:, 1], c[:, 0], '.', color=colors[j], zorder=2)
            #
            #     print 'Cluster: {}, group: {}. ratio:{}. length:{}: bbox:{}'.format(i, g, ratio, np.shape(c),
            #                                                                         (x1, x2, y1, y2))
    # print (x1, y1), x2 - x1, y2 - y1
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=lw, edgecolor=color, facecolor='none',
                             zorder=zorder)

    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20


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


def visualise_clusters(data, true_tracks, leaves, filename, limits=None, debug=False, pub=False):
    if debug:
        # plt.clf()
        cluster_labels = data[:, 5]
        unique_labels = np.unique(cluster_labels)
        fig, ax = plt.subplots(1)

        for g in unique_labels:
            c = np.vstack(data[cluster_labels == g, 0])
            ax.plot(c[:, 1], c[:, 0], '.')

            if not pub:
                ax.annotate(g, (np.mean(c[:, 1]), np.mean(c[:, 0])))

            # print 'cluster {}: inertia ratio is: {}'.format(g, __ir2(c, 1000, g))

        for i, (cluster, j, msg, bbox) in enumerate(leaves):
            x1, x2, y1, y2, = bbox
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='gray', facecolor='none',
                                     zorder=-1)

            # Plot mean cluster
            mx = np.average(cluster[:, 1], weights=cluster[:, 2])
            my = np.average(cluster[:, 0], weights=cluster[:, 2])
            ax.plot(mx, my, '*', zorder=10, color='w', markeredgecolor='k',
                    markersize=7.5)

            # if not pub:
            ax.text(x1 + 0.95 * (x2 - x1), y1 + 0.95 * (y2 - y1), j, color='k', fontsize=8, ha='right', va='top',
                    zorder=10)
            # Add the patch to the Axes
            ax.add_patch(rect)

        if true_tracks:
            ax = visualise_true_tracks(ax, true_tracks)

        ax.set(xlabel='Sample', ylabel='Channel')
        ax = set_limits(ax, limits)
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        plt.grid()
        ax.figure.savefig(filename)


def visualise_candidates(candidates, true_tracks, filename, limits=None, debug=False):
    if debug:
        # plt.clf()
        fig, ax = plt.subplots(1)
        for c in candidates:
            group_2 = c[:, 3][0]
            ax = sns.scatterplot(x=c[:, 1], y=c[:, 0])
            ax.annotate(group_2, (np.mean(c[:, 1]), np.mean(c[:, 0])))

        ax = visualise_true_tracks(ax, true_tracks)

        ax.set(xlabel='Sample', ylabel='Channel', title='Detected Candidates')
        ax = set_limits(ax, limits)
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20

        ax.figure.savefig(filename)


def visualise_tracks(tracks, true_tracks, filename, limits=None, debug=False, pub=False):
    if debug:
        # plt.clf()
        fig, ax = plt.subplots(1)
        for c in tracks:
            group_2 = c[:, 3][0]
            ax.plot(c[:, 1], c[:, 0], '.')
            if not pub:
                ax.annotate(group_2, (np.mean(c[:, 1]), np.mean(c[:, 0])))

        ax = visualise_true_tracks(ax, true_tracks)

        ax.set(xlabel='Sample', ylabel='Channel')
        ax = set_limits(ax, limits)
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        plt.tight_layout()
        ax.grid(b=True, which='major', color='#e0e0e0', linestyle='--')
        ax.figure.savefig(filename)


def visualise_post_processed_tracks(tracks, true_tracks, filename, limits=None, groups=None, debug=False):
    if debug:
        # plt.clf()
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

            score = (abs(r_value) + abs(1 - p) + abs(1 - e)) / 3
            param = c[:, 1]
            missing = np.setxor1d(np.arange(min(param), max(param)), param)
            score = len(missing) / float(len(param))
            print(group,
                  'S:{:0.5f} R:{:0.5f} P:{:0.5f} E:{:0.5f} I:{:0.5f} N:{} M:{:0.4f} C:{:0.4f} X:{:0.4f} Y:{:0.4f} SNR:{:0.2f}'
                  .format(score, r_value, p, e, ratio[0], len(c), m, intercept, x, y, np.mean(c[:, 2])))

            ax.annotate('{:0.0f}'.format(group), (x, 1.01 * y), zorder=3)
            if groups:
                if group not in groups:
                    continue

            missing = c[c[:, 3] == -2]
            thickened = c[c[:, 3] == -3]
            detected = c
            ax = sns.scatterplot(detected[:, 1], detected[:, 0], color='green', marker=".", zorder=4)
            ax = sns.scatterplot(thickened[:, 1], thickened[:, 0], marker=".", color='pink', zorder=2, edgecolor="k")
            ax = sns.scatterplot(missing[:, 1], missing[:, 0], marker="+", color='red', zorder=3, edgecolor="k")

        ax = visualise_true_tracks(ax, true_tracks)

        ax.set(xlabel='Sample', ylabel='Channel')
        ax = set_limits(ax, limits)
        plt.grid()
        ax.figure.savefig(filename)


def visualise_filtered_data(ndx, true_tracks, filename, limits=None, debug=False, pub=False):
    if debug:
        fig, ax = plt.subplots(1)
        if limits:
            x_start, x_end, y_start, y_end = limits
            ndx = _partition(ndx, x_start, x_end, y_start, y_end)
        ax.scatter(ndx[:, 1], ndx[:, 0], color='gray', marker='.')
        # ax = sns.scatterplot(x=ndx[:, 1], y=ndx[:, 0], marker='.', color='r', zorder=1)

        ax = set_limits(ax, limits)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15

        if true_tracks:
            ax = visualise_true_tracks(ax, true_tracks)
        ax.set(xlabel='Sample', ylabel='Channel')
        ax = set_limits(ax, limits)

        ax.grid(b=True, which='major', color='#e0e0e0', linestyle='--')
        plt.tight_layout()
        ax.figure.savefig(filename)


def visualise_tree_traversal(ndx, true_tracks, clusters, leaves, filename, limits=None, vis=False, pub=False):
    if vis:
        # plt.clf()
        fig, ax = plt.subplots(1)
        x_start, x_end, y_start, y_end = 0, ndx.shape[0], 0, ndx.shape[1]
        if limits:
            x_start, x_end, y_start, y_end = limits
            ndx = _partition(ndx, x_start, x_end, y_start, y_end)

        ax.plot(ndx[:, 1], ndx[:, 0], 'r.')

        ax = set_limits(ax, limits)

        # print  x_start, x_end, y_start, y_end
        for leaf_id, (cluster, bbox) in enumerate(leaves):
            # cluster_id = i
            x1, x2, y1, y2 = bbox
            if x_start * 0.4 <= x1 <= x_end * 1.5 and y_start * 0.5 <= y1 <= y_end * 1.5:
                # if j != 0:
                #     cluster_id = j
                __plot_leaf(ax, x1, y1, x2, y2, leaf_id, '', False, positives=None, negatives=cluster, pub=pub)

        for i, (cluster, cluster_id, ratio, bbox) in enumerate(clusters):
            # cluster_id = i
            x1, x2, y1, y2, = bbox
            if x_start * 0.4 <= x1 <= x_end * 1.5 and y_start * 0.5 <= y1 <= y_end * 1.5:
                # if j != 0:
                #     cluster_id = j
                __plot_leaf(ax, x1, y1, x2, y2, cluster_id, ratio, True, positives=cluster, negatives=None, pub=pub)

        if true_tracks:
            ax = visualise_true_tracks(ax, true_tracks)
        ax.set(xlabel='Sample', ylabel='Channel')
        ax = set_limits(ax, limits)

        plt.tight_layout()
        ax.grid(b=False)

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

    print(metrics_df[['precision', 'recall', 'snr', 'f1']])


def compare_algorithms(db_scan_clusters, msds_clusters, iteration, limits=None,
                       debug=False):
    if debug:
        # plt.clf()
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
        time_samples = 160

        ax1 = axes[0]
        ax1.set_title('DBSCAN')
        for i, cluster in enumerate(db_scan_clusters):
            c = cluster.data
            ax1.plot(c['time_sample'] - time_samples * iteration, c['channel_sample'], 'g.', zorder=4, ms=2)

            # print iteration, 'DBSCAN: Cluster {}, N: {}, Unique Samples: {}, Unique Channels: {} Beam:{}'.format(
            #     i, len(c['time_sample']), len(np.unique(c['time_sample'])), len(np.unique(c['channel_sample'])), c['beam_id'][0])

        ax2 = axes[1]
        ax2.set_title('MSDS')
        for i, cluster in enumerate(msds_clusters):
            c = cluster.data
            ax2.plot(c['time_sample'] - time_samples * iteration, c['channel_sample'], 'r.', zorder=4, ms=2)

            # print 'channel_sample', np.mean(c['channel_sample'])
            # print iteration, 'MSDS: Cluster {}, N: {}, Unique Samples: {}, Unique Channels: {}'.format(
            #     i, len(c['time_sample']), len(np.unique(c['time_sample'])), len(np.unique(c['channel_sample'])))

        ax1.set(xlabel='Sample', ylabel='Channel', xticks=range(0, 160, 40))
        ax2.set(xlabel='Sample', xticks=range(0, 160, 40))
        plt.grid()

        ax2.figure.savefig('msds_dbscan_comparison_{}.png'.format(iteration))

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


def plot_metric(metrics_df, metric, file_name=None):
    fig, ax = plt.subplots(1)
    algorithms = metrics_df.index.unique()

    for name in algorithms:
        if name.endswith('_pp'):
            continue
        data = metrics_df[metrics_df.index == name]
        ax.plot(data['snr'], data[metric], '-', lw=3, label='{}'.format(name.title()))

    ax.set(xlabel='SNR (dB)', ylabel=metric.title())

    plt.grid(alpha=0.3)

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(prop=fontP, bbox_to_anchor=(1.04, 1.00), borderaxespad=0)

    if file_name:
        plt.tight_layout()
        fig.savefig(file_name)


def mono_cylcer():
    return (cycler('color', ['k']) * cycler('markerfacecolor', ['w']) * cycler('markeredgecolor', ['k']) * cycler(
        'linestyle', ['-']) * cycler('marker', ['o', '+', 'x', '^', '.', '']))
    # return (cycler('color', ['k', 'gray']) * cycler('linestyle', ['-', '--', ':', '=.']) * cycler('marker', ['^', ',', '.']))


def plot_metric_combined(metrics_df, metrics, pp_results=False, file_name=None):
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    algorithms = metrics_df['name'].unique()
    monochrome = mono_cylcer()

    for i, ax in enumerate(axes):
        metric = metrics[i]
        ax.set_prop_cycle(monochrome)
        for name in algorithms:
            if pp_results:
                if not name.endswith('_pp'):
                    continue
                name = name[:-3]
            else:
                if name.endswith('_pp'):
                    continue
            data = metrics_df[metrics_df['name'] == name]

            ax.plot(data['snr'], data[metric]['mean'], lw=1, label='{}'.format(name.title()))

        ax.set(ylabel=metric.title())

        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.tick_params(width=2)

    # axes[1].yaxis.set_ticks(np.arange(0.7, 1.0, 0.1))

    from matplotlib.ticker import FormatStrFormatter
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    lgd = axes[0].legend(bbox_to_anchor=(1.0, 1.05), frameon=False)
    axes[2].set(xlabel="SNR (dB)")

    if file_name:
        # plt.subplots_adjust(right=1.15)
        plt.tight_layout()
        fig.savefig(file_name, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')


def plot_timings(metrics_df, algorithms, include_pp=False, file_name=None):
    fig, ax = plt.subplots(1)
    # algorithms = metrics_df.index.unique()
    errors_kw = dict(capsize=9, capthick=3, elinewidth=3, zorder=4, ecolor='#666666')

    metrics_df = metrics_df[metrics_df['snr'] == metrics_df['snr'].max()]
    for name in algorithms:
        name_str = name.title().replace('_', ' ')
        data = metrics_df[metrics_df['name'] == name]
        f_time = data['dt']['mean']
        b1 = ax.barh(name_str, f_time, xerr=data['dt']['std'], color='#e0e0e0', zorder=3, edgecolor='#666666',
                     alpha=0.8, error_kw=errors_kw)

        if include_pp:
            data = metrics_df[metrics_df['name'] == name + '_pp']
            b2 = ax.barh(name_str, data['dt']['mean'], xerr=data['dt']['std'], left=f_time, color='#999999', zorder=3,
                         edgecolor='#666666', alpha=0.8, error_kw=errors_kw)

    # plt.legend((b1[0], b2[0]), ('Men', 'Women'))
    ax.invert_yaxis()
    ax.set(xlabel='Processing Time (s)')
    ax.grid(which='major')

    ax.set_axisbelow(True)
    plt.rc('axes', labelsize=45)
    ax.xaxis.labelpad = 15
    if file_name:
        plt.tight_layout()
        fig.savefig(file_name)


def plot_timings_detector(metrics_df, algorithms, file_name=None):
    fig, ax = plt.subplots(1)
    # algorithms = metrics_df.index.unique()
    errors_kw = dict(capsize=9, capthick=3, elinewidth=3, zorder=4, ecolor='#666666')
    # snrs = [1, 5, 10]

    u = metrics_df['snr'].unique()
    snrs = [u.min(), np.argsort(u)[len(u) // 2], u.max()]
    snrs = [1.0, 3.0, 7.]

    bw = 0.4
    left = bw / 2.
    ng = len(algorithms)
    l = [bw + i * bw * (ng + 1) for i in np.arange(ng)]
    # algorithms = ['Astride', 'DBSCAN', 'MSDS', 'Hough']
    algorithms = list(metrics_df['name'].unique())
    for i, name in enumerate(algorithms):
        for j, snr in enumerate(snrs):
            data = metrics_df[metrics_df['snr'] == snr][metrics_df['name'] == name]
            f_time = data['dt']['mean']

            b1 = ax.bar(left, f_time, yerr=data['dt']['std'], zorder=3, edgecolor='#666666',
                        alpha=0.61, error_kw=errors_kw, label=snr, width=bw, color=COLORS[j])

            left += bw

        left += bw

        # legend['labels'].append('SNR = {} dB'.format(snr))

    plt.xticks([0.6, 2.2, 3.8, 5.4], algorithms)
    plt.legend(['SNR = {:1.0f} dB'.format(snr) for snr in snrs])

    # ax.invert_yaxis()
    ax.set(xlabel='Processing Time (s)')
    ax.grid(which='major')

    ax.set_axisbelow(True)
    plt.rc('axes', labelsize=45)
    ax.xaxis.labelpad = 15
    if file_name:
        plt.tight_layout()
        fig.savefig(file_name)


def viz_den(X, Z, threshold, labels):
    from scipy.cluster import hierarchy
    from matplotlib.ticker import MaxNLocator
    hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
    fig, (ax1, ax2) = plt.subplots(2, 1)

    dn1 = hierarchy.dendrogram(Z, ax=ax1, count_sort=True, labels=X[:, 1].astype(int), above_threshold_color='y',
                               color_threshold=2, orientation='top')
    hierarchy.set_link_color_palette(None)  # reset to default after use

    u_labels = np.unique(labels)
    for u in u_labels:
        ax2.plot(X[labels == u][:, 1].astype(int), X[labels == u][:, 0], '.', ms=15)

    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()


def plot_TLE(obs_info, input_data, tle_target, beam_id=15):
    expected_transit_time = tle_target.transit_time
    target_name = tle_target.name
    expected_doppler = tle_target.doppler

    start_window = expected_transit_time - datetime.timedelta(seconds=20)
    end_window = expected_transit_time + datetime.timedelta(seconds=10)

    print('Current time is', obs_info['timestamp'])
    print('TLE is between {} and {}'.format(start_window, end_window))
    print('Persister Within window', end_window >= obs_info['timestamp'] >= start_window, obs_info['iter_count'])

    if end_window >= obs_info['timestamp'] >= start_window:
        expected_channel = (obs_info['transmitter_frequency'] * 1e6 + expected_doppler) - obs_info[
            'start_center_frequency'] * 1e6
        expected_channel /= obs_info['channel_bandwidth'] * 1e6
        doppler_offset = 2392
        expected_channel = int(expected_channel) - doppler_offset

        channel_window = expected_channel - 100, expected_channel + 100

        filename = '{}_TLE_prediction_{}'.format(target_name, obs_info['iter_count'])

        time = [obs_info['timestamp'], datetime.timedelta(seconds=obs_info['sampling_time'] * 160) + obs_info[
            'timestamp']]

        subset = input_data[beam_id, channel_window[0]: channel_window[1], 120:]

        print('Expecting target to be between', channel_window, subset.shape)
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Channel")
        im = ax.imshow(subset, aspect='auto', interpolation='none', origin='lower', vmin=0,
                       extent=[120, 160, channel_window[0], channel_window[1]])
        fig.colorbar(im)
        fig.tight_layout()
        plt.title(filename + ' from {:%H.%M.%S} to {:%M.%S}'.format(time[0], time[1]))
        plt.show()

        fig.savefig(filename, bbox_inches='tight')


def plot_RSO_track(track, i, beam_id=None):
    fig, ax = plt.subplots(1)

    mask = track.data['beam_id'] > -1
    if beam_id:
        mask = track.data['beam_id'] == beam_id

    time = track.data[mask]['time_sample']
    channel = track.data[mask]['channel_sample']
    ax.plot(time, channel, 'g.')

    reduced_track = track.reduce_data()

    mask = reduced_track['beam_id'] > -1
    if beam_id:
        mask = reduced_track['beam_id'] == beam_id

    t1 = reduced_track[mask]['time_sample']
    c1 = reduced_track[mask]['channel_sample']
    ax.plot(t1, c1, 'r+')

    mean_x = np.mean(time)
    mean_y = np.mean(channel)
    ax.plot(mean_x, mean_y, '*')
    ax.set(xlabel='Sample', ylabel='Channel')

    ax.annotate('RSO {}'.format(i), (np.mean(time), np.mean(channel)))

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    plt.grid()

    ax.figure.savefig('{}.png'.format(i))


def plot_RSO_track_snr(track, i, beam_id=None):
    fig, ax = plt.subplots(1)

    mask = track.data['beam_id'] > -1
    if beam_id:
        mask = track.data['beam_id'] == beam_id

    time = track.data[mask]['time_sample']
    snr = track.data[mask]['snr']
    ax.plot(time, snr, '.')

    reduced_track = track.reduce_data()

    if beam_id:
        mask = reduced_track['beam_id'] == beam_id
        t1 = reduced_track[mask]['time_sample']
        c1 = reduced_track[mask]['snr']
        ax.plot(t1, c1, 'r+')
    else:
        beams = np.unique(reduced_track['beam_id'])
        for b in beams:
            mask = reduced_track['beam_id'] == b
            t1 = reduced_track[mask]['time_sample']
            c1 = reduced_track[mask]['snr']
            ax.plot(t1, c1, '-')

    mean_x = np.mean(time)
    mean_y = np.mean(snr)
    ax.plot(mean_x, mean_y, '*')
    ax.set(xlabel='Sample', ylabel='SNR')

    ax.annotate('RSO {}'.format(i), (np.mean(time), np.mean(snr)))

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    plt.grid()

    ax.figure.savefig('{}.png'.format(i))


def plot_all_RSOs_track(tracks):
    fig, ax = plt.subplots(1)

    for i, track in enumerate(tracks):
        time = track.data['time_sample']
        channel = track.data['channel_sample']
        ax.plot(time, channel, '.')
        ax.set(xlabel='Sample', ylabel='Channel')
        mean_x = np.mean(time)
        mean_y = np.mean(channel)
        ax.plot(mean_x, mean_y, '*', markersize=20, zorder=10)

        ax.annotate('{}'.format("{}/{}".format(i + 1, len(tracks))), (np.mean(time), np.mean(channel)))

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    plt.grid()
