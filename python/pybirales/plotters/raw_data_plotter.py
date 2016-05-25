import math
from matplotlib import pyplot as plt

import logging
import numpy as np

from pybirales.base.definitions import PipelineError
from pybirales.base.plotter import Plotter
from pybirales.blobs.dummy_data import DummyBlob
from pybirales.blobs.receiver_data import ReceiverBlob


class RawDataPlotter(Plotter):
    def __init__(self, config, input_blob, figure):

        # Make sure that the input data blob is what we're expecting
        if type(input_blob) not in [DummyBlob, ReceiverBlob]:
            raise PipelineError("RawDataPlotter: Invalid input data type, should be DummyBlob or ReceiverBlob")

        # Call superclass initialiser
        super(RawDataPlotter, self).__init__(config, input_blob, figure)

        # Get dimensions from input data blob
        input_shape = dict(self._input.shape)
        self._nants = input_shape['nants']
        self._nsamp = input_shape['nsamp']

        # Figure and axes placeholders
        self._antennas_to_plot = None

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

        return slice(None), slice(nof_samples), slice(self._antennas_to_plot[0], self._antennas_to_plot[-1] + 1)

    def initialise_plot(self):
        """ Initialise plot """

        # Initialise figure
        plt.title("Antenna Plot")
        self._figure.set_tight_layout(0.9)

    def refresh_plot(self, input_data, obs_info):
        """ Update plot with new data
        :param input_data: Input data to plot
        :param obs_info: Observation information """

        self._figure.clf()
        plt.title("Antenna plot")
        for index, ant in enumerate(self._antennas_to_plot):
            plt.plot(np.abs(input_data[0, :, index]), label="Antenna %d" % ant)
            plt.xlabel("Time")
            plt.ylabel("Power")
            plt.xlim([0, input_data.shape[1]])

        plt.legend()
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
        plt.show(block=False)

        logging.info("RawDataPlotter: Updated plot")
