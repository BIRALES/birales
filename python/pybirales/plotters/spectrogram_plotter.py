import matplotlib.pyplot as plt
import numpy as np
import logging as log
from matplotlib import colors
from pybirales.base import settings
import copy


class SpectrogramPlotter:
    _plot_dir = 'plots/'

    def __init__(self):
        self.fig = plt.figure()
        self.colors = [
            'g', 'b', 'y', 'c', 'r', 'k', 'm'
        ]

        self._data = None
        self._last_id = None
        self._count = 0
        self._freq = 10

    def plot(self, data, filename, condition=True, cluster_labels=None):
        if not settings.detection.debug_candidates:
            return

        if condition:
            try:
                if self._count % self._freq == 0:
                    if self._count != 0:
                        plt.title(filename)
                        plt.imshow(self._data, aspect='auto', interpolation='none', origin='lower',  vmin=0)
                        plt.colorbar()
                        if cluster_labels is not None:
                            categories = [self.colors[c] for c in cluster_labels if c in self.colors]
                            d = np.column_stack(np.where(data > 0))
                            # plt.scatter(d[:, 1], d[:, 0], c=categories)

                        self.fig.savefig(self._plot_dir + filename + '_' + str(self._count) + '.png', bbox_inches='tight')
                        self.fig.clf()
                    self._data = copy.deepcopy(data)
                else:
                    self._data = np.vstack((self._data, copy.deepcopy(data)))

                self._count += 1


            except Exception as e:
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
