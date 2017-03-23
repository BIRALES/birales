import matplotlib.pyplot as plt
import numpy as np
import logging as log


class SpectrogramPlotter:
    _plot_dir = 'plots/'

    def __init__(self):
        self.fig = plt.figure()

    def plot(self, data, filename, condition=True):
        if condition:
            try:
                self.fig.clf()
                plt.imshow(data, aspect='auto', interpolation='none', origin='lower')

                self.fig.savefig(self._plot_dir + filename + '.png')
            except Exception:
                exit()

    def scatter(self, x, y, filename, condition=True):
        if condition:
            try:
                self.fig.clf()
                plt.plot(x[:100], "-")
                self.fig.savefig(self._plot_dir + filename + '.png')
            except Exception:
                exit()
plotter = SpectrogramPlotter()
