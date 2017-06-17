import matplotlib.pyplot as plt
import numpy as np
import logging as log
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
        self._freq = 5
        self._min_channel = 2500
        self._max_channel = 3000

    def _file_path(self, filename):
        min_count = self._count - self._freq
        return self._plot_dir + filename + '_' + str(min_count) + '-' + str(self._count) + '.png'

    def plot(self, beam, filename, condition=True):
        if not settings.detection.debug_candidates:
            return

        if condition:
            if self._count % self._freq == 0:
                if self._count != 0:
                    plt.title(filename + '_' + str(self._count - self._freq) + '-' + str(self._count))
                    plt.imshow(self._data, aspect='auto', interpolation='none', origin='lower', vmin=0)
                    plt.colorbar()

                    self.fig.savefig(self._file_path(filename), bbox_inches='tight')
                    self.fig.clf()
                    log.info('Saved %s', self._file_path(filename))
                self._data = copy.deepcopy(beam.snr[:, self._min_channel:self._max_channel])
            else:
                self._data = np.vstack((self._data, np.ones((1, self._max_channel - self._min_channel))))
                self._data = np.vstack((self._data, copy.deepcopy(beam.snr[:, self._min_channel:self._max_channel])))
            self._count += 1

    def plot_detections(self, beam, filename, condition, clusters):
        if not settings.detection.debug_candidates:
            return

        if condition:
            if np.any(clusters):
                for cluster in clusters:
                    beam.snr[cluster.indices[0], cluster.indices[1]] = 50
            if self._count % self._freq == 0:
                if self._count != 0:
                    plt.title(filename + '_' + str(self._count - self._freq) + '-' + str(self._count))
                    plt.imshow(self._data, aspect='auto', interpolation='none', origin='lower')
                    plt.colorbar()

                    self.fig.savefig(self._file_path(filename), bbox_inches='tight')
                    self.fig.clf()
                    log.info('Saved %s', self._file_path(filename))
                self._data = copy.deepcopy(beam.snr[:, self._min_channel:self._max_channel])
            else:
                beam.snr[0, self._min_channel:self._max_channel] = 1  # divider
                self._data = np.vstack((self._data, copy.deepcopy(beam.snr[:, self._min_channel:self._max_channel])))
            self._count += 1


plotter = SpectrogramPlotter()
