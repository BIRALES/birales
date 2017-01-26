import logging as log
import os
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np

# from pybirales.configuration.application import config


class Plotter:
    def __init__(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def save(self, file_path):
        pass


class MultiLineMatplotlibPlotter(Plotter):
    def __init__(self, fig_size, fig_title, plot_title, x_limits, y_limits, x_label, y_label, data):
        Plotter.__init__(self)

        self.n_cols = 4
        self.n_rows = len(data) / self.n_cols
        self.fig_size = fig_size
        self.fig_title = fig_title
        self.plot_title = plot_title
        self.x_lim = x_limits
        self.y_lim = y_limits
        self.x_lim_n = 4
        self.y_lim_n = 8
        self.x_label = x_label
        self.y_label = y_label
        self.data = data

    def plot(self):
        fig = self._build()
        fig.show()

    def save(self, file_name):
        file_path = os.path.join(self.output_dir, file_name)
        if not os.path.isfile(file_path):
            fig = self._build()
            fig.savefig(file_path)
            log.debug('%s visualisation saved in %s', file_name, file_path)
        return file_path

    def _build(self):
        fig, axs = plt.subplots(nrows=self.n_rows, ncols=self.n_cols, sharex=True, sharey=True, figsize=self.fig_size)

        fig.suptitle(self.fig_title, fontsize=12)
        index = 0
        n_plots = len(self.data)
        for row in range(0, self.n_rows):
            for col in range(0, self.n_cols):
                if n_plots < (col * row):
                    break

                y_data = self.data[index]

                x_axis = np.linspace(self.x_lim[0], self.x_lim[1], self.x_lim_n)
                y_axis = np.linspace(self.y_lim[0], self.y_lim[1], self.y_lim_n)

                ax = self._get_grid_position(axs, col, row)
                ax.set_title(self.plot_title + ' ' + str(index), fontsize=7, y=0.95)
                ax.set_xticks(x_axis)
                ax.set_yticks(y_axis)

                ax.tick_params(axis='both', which='major', labelsize=8)

                ax.plot(y_data)
                index += 1

        fig.text(0.5, 0.04, self.x_label, ha='center', va='center')
        fig.text(0.06, 0.5, self.y_label, ha='center', va='center', rotation='vertical')

        return fig

    def _get_grid_position(self, axs, col, row):
        if self.n_rows > 1:
            ax = axs[row, col]
        else:
            ax = axs[col]
        return ax


class MultiWaterfallMatplotlibPlotter(Plotter):
    def __init__(self, fig_size, fig_title, plot_title, x_limits, y_limits, x_label, y_label, data):
        Plotter.__init__(self)

        self.n_cols = 2
        self.n_rows = len(data) / self.n_cols
        self.fig_size = fig_size
        self.fig_title = fig_title
        self.plot_title = plot_title
        self.x_lim = x_limits
        self.y_lim = y_limits
        self.x_lim_n = 4
        self.y_lim_n = 8
        self.x_label = x_label
        self.y_label = y_label
        self.data = data

    def plot(self):
        fig = self._build()
        fig.show()

    def save(self, file_name):
        file_path = os.path.join(self.output_dir, file_name)
        if not os.path.isfile(file_path):
            fig = self._build()
            fig.savefig(file_path)
            log.debug('%s visualisation saved in %s', file_name, file_path)
        return file_path

    def _build(self):
        fig, axs = plt.subplots(nrows=self.n_rows, ncols=self.n_cols, sharex=True, sharey=True, figsize=self.fig_size)
        fig.suptitle(self.fig_title, fontsize=12)
        n_plots = len(self.data)
        index = 0
        for row in range(0, self.n_rows):
            for col in range(0, self.n_cols):
                if n_plots < (col * row):
                    break

                y_data = self.data[index]

                x_axis = np.linspace(self.x_lim[0], self.x_lim[1], self.x_lim_n)
                y_axis = np.linspace(self.y_lim[0], self.y_lim[1], self.y_lim_n)

                ax = self._get_grid_position(axs, col, row)
                ax.set_title(self.plot_title + ' ' + str(index), fontsize=7, y=0.99)
                ax.set_xticks(x_axis)
                ax.set_yticks(y_axis)

                ax.tick_params(axis='both', which='major', labelsize=8)

                ax.imshow(y_data, aspect='auto')
                index += 1

        fig.text(0.5, 0.04, self.x_label, ha='center', va='center')
        fig.text(0.06, 0.5, self.y_label, ha='center', va='center', rotation='vertical')

        return fig

    def _get_grid_position(self, axs, col, row):
        if self.n_rows > 1:
            ax = axs[row, col]
        else:
            ax = axs[col]
        return ax


class BeamMatplotlibPlotter(Plotter):
    # Figure size in inches
    _fig_size = (16, 9)

    # Limits of the color map to aid in manual verification
    _c_lim = (0.3, 0.7)

    def __init__(self, fig_size, fig_title, plot_title, x_limits, y_limits, x_label, y_label, data):
        Plotter.__init__(self)

        self.fig_size = fig_size
        self.fig_title = fig_title
        self.plot_title = plot_title
        self.x_lim = x_limits
        self.y_lim = y_limits
        self.x_lim_n = 4
        self.y_lim_n = 8
        self.x_label = x_label
        self.y_label = y_label
        self.data = data

        self.file_name = plot_title.replace(' ', '_') + config.get('monitoring', 'IMAGE_EXT')

    def plot(self):
        fig = self._build()

        plt.show()

    def save(self, base_dir):
        file_path = os.path.join(base_dir, self.file_name)
        fig = self._build()
        fig.savefig(file_path)
        log.debug('%s visualisation saved in %s', self.file_name, file_path)

        return file_path

    def _build(self):
        fig = plt.figure(figsize=self._fig_size)

        ax = fig.add_subplot(1, 1, 1)

        if self.x_lim is not 'auto':
            x_axis = np.linspace(self.x_lim[0], self.x_lim[1], self.x_lim_n)
            ax.set_xticks(x_axis)

        if self.y_lim is not 'auto':
            y_axis = np.linspace(self.y_lim[0], self.y_lim[1], self.y_lim_n)
            ax.set_yticks(y_axis)

        ax.set_title(self.plot_title)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

        im = ax.imshow(self.data,
                       aspect='auto',
                       origin='lower',  clim=self._c_lim)

        fig.colorbar(im, ax=ax)

        return fig
