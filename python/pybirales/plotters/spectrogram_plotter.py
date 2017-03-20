import matplotlib.pyplot as plt
import numpy as np


class SpectrogramPlotter:
    _plot_dir = 'plots/'

    def __init__(self):
        self.fig = plt.figure()

    def plot(self, data, filename, condition=True):
        if condition:
            self.fig.clf()
            plt.imshow(data, aspect='auto', interpolation='none', origin='lower')
            self.fig.savefig(self._plot_dir + filename + '.png')

    def scatter(self, x, y, filename, condition=True):
        if condition:
            self.fig.clf()
            plt.plot(np.abs(x[:100]), "-")
            self.fig.savefig(self._plot_dir + filename + '.png')

plotter = SpectrogramPlotter()
