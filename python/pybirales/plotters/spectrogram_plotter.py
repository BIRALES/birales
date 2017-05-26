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

    def _file_path(self, filename):
        min_count = self._count - self._freq
        return self._plot_dir + filename + '_' + str(min_count) + '-' + str(self._count) + '.png'

    def plot(self, beam, filename, condition=True):
        if not settings.detection.debug_candidates:
            return

        if condition:
            if self._count % self._freq == 0:
                if self._count != 0:
                    plt.title(filename)
                    plt.imshow(self._data, aspect='auto', interpolation='none', origin='lower',  vmin=0)
                    plt.colorbar()

                    self.fig.savefig(self._file_path(filename), bbox_inches='tight')
                    self.fig.clf()
                self._data = copy.deepcopy(beam.snr)
            else:
                self._data = np.vstack((self._data, copy.deepcopy(beam.snr)))
            self._count += 1

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
