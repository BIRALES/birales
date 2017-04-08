import matplotlib.pyplot as plt
import numpy as np
import logging as log
from matplotlib import colors
from pybirales.base import settings


class SpectrogramPlotter:
    _plot_dir = 'plots/'

    def __init__(self):
        self.fig = plt.figure()
        self.colors = [
            'g', 'b', 'y', 'c', 'r', 'k', 'm'
        ]

    def plot(self, data, filename, condition=True, cluster_labels=None):
        if not settings.detection.debug_candidates:
            return

        if cluster_labels is not None:
            categories = [self.colors[c] for c in cluster_labels if c in self.colors]

        if condition:
            try:
                self.fig.clf()
                plt.imshow(data, aspect='auto', interpolation='none', origin='lower')
                if cluster_labels is not None:
                    d = np.column_stack(np.where(data > 0))
                    plt.scatter(d[:, 1], d[:, 0], c=categories)
                self.fig.savefig(self._plot_dir + filename + '.png')
            except Exception:
                log.exception('An error has occurred. Quiting')
                exit()

    def scatter(self, x, y, filename, condition=True):
        if not settings.detection.debug_candidates:
            return

        if condition:
            try:
                self.fig.clf()
                plt.plot(x[:100], "-")
                self.fig.savefig(self._plot_dir + filename + '.png')
            except Exception:
                exit()


plotter = SpectrogramPlotter()
