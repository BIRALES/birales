import math
from matplotlib import pyplot as plt

import logging
import numpy as np

from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.plotter import Plotter
from pybirales.pipeline.blobs.dummy_data import DummyBlob
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob


class RawDataGridPlotter(Plotter):
    def __init__(self, config, input_blob, figure):

        # Make sure that the input data blob is what we're expecting
        if type(input_blob) not in [DummyBlob, ReceiverBlob]:
            raise PipelineError("RawDataGridPlotter: Invalid input data type, should be DummyBlob or ReceiverBlob")

        # Call superclass initialiser
        super(RawDataGridPlotter, self).__init__(config, input_blob, figure)

        # Get dimensions from input data blob
        input_shape = dict(self._input.shape)
        self._nants = input_shape['nants']
        self._nsamp = input_shape['nsamp']

        # Figure and axes placeholders
        self._antennas_to_plot = None
        self._axes = None

    def create_index(self):
        """ Create data index to get from blob """

        # Check whether config file has any indexing defined
        nof_samples = None
        self._antennas_to_plot = range(self._nants)
        if self._config is not None:
            if 'antenna_range' in self._config.settings():
                if type(self._config.antenna_range) is list:
                    self._antennas_to_plot = range(self._config.antenna_range[0], self._config.antenna_range[1] + 1)
                else:
                    self._antennas_to_plot = range(self._config.antenna_range, self._config.antenna_range + 1)
            if 'nof_samples' in self._config.settings():
                nof_samples = self._config.nof_samples

        return slice(0, 1, None), slice(None), slice(nof_samples), slice(self._antennas_to_plot[0], self._antennas_to_plot[-1] + 1)

    def initialise_plot(self):
        """ Initialise plot """

        # Add a subplot per antenna
        self._axes = []

        if len(self._antennas_to_plot) == 2:
            axes = plt.subplot2grid((1, 2), (0, 0))
            plt.title("Antenna %d" % self._antennas_to_plot[0])
            self._axes.append(axes)
            axes = plt.subplot2grid((1, 2), (0, 1))
            plt.title("Antenna %d" % self._antennas_to_plot[1])
            self._axes.append(axes)
        else:
            grid_dim_x = int(math.ceil(math.sqrt(len(self._antennas_to_plot))))
            grid_dim_y = int(math.ceil(math.sqrt(len(self._antennas_to_plot))))

            for index, i in enumerate(self._antennas_to_plot):
                axes = plt.subplot2grid((grid_dim_x, grid_dim_y),
                                        (int((index / grid_dim_x)),
                                         int(index % grid_dim_x)))
                plt.title("Anetnna %d" % i)
                self._axes.append(axes)

    def refresh_plot(self, input_data, obs_info):
        """ Update plot with new data
        :param input_data: Input data to plot
        :param obs_info: Observation information """

        # Loop over all antennas to plot
        for index, antenna in enumerate(self._antennas_to_plot):
            self._axes[index].cla()
            self._axes[index].plot(np.abs(input_data[0, 0, :, index]))
            self._axes[index].set_xlim([0, input_data.shape[2]])
            self._axes[index].set_xlabel("Time")
            self._axes[index].set_ylabel("Channel")
            self._axes[index].set_title("Antenna %d" % antenna)

        self._figure.subplots_adjust(right=0.9)
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
        plt.show(block=False)
