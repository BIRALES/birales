import logging
import numpy as np

from matplotlib import pyplot as plt

from pybirales.base.definitions import PipelineError
from pybirales.base.plotter import Plotter
from pybirales.blobs.receiver_data import ReceiverBlob


class AntennaPlotter(Plotter):
    def __init__(self, config, input_blob, figure):

        # Make sure that the input data blob is what we're expecting
        if type(input_blob) is not ReceiverBlob:
            raise PipelineError("AntennaPlotter: Invalid input data type, should be ChannelisedBlob")

        # Call superclass initialiser
        super(AntennaPlotter, self).__init__(config, input_blob, figure)

        # Get dimensions from input data blob
        input_shape = dict(self._input.shape)
        self._nants = input_shape['nants']
        self._nsubs = input_shape['nsubs']

        # Figure and axes placeholders
        self._subbands_to_plot = None
        self._antennas_to_plot = None

    def create_index(self):
        """ Create data index to get from blob """

        # Check whether config file has any indexing defined
        nof_samples = None
        self._antennas_to_plot = range(self._nants)
        self._subbands_to_plot = range(self._nsubs)

        if self._config is not None:
            if 'subband_range' in self._config.settings():
                if type(self._config.subband_range) is list:
                    self._subbands_to_plot = range(self._config.subband_range[0], self._config.subband_range[1] + 1)
                else:
                    self._subbands_to_plot = range(self._config.subband_range, self._config.subband_range + 1)

            if 'antenna_range' in self._config.settings():
                if type(self._config.antenna_range) is list:
                    self._antennas_to_plot = range(self._config.antenna_range[0], self._config.antenna_range[1] + 1)
                else:
                    self._antennas_to_plot = range(self._config.antenna_range, self._config.antenna_range + 1)

            if 'nof_samples' in self._config.settings():
                nof_samples = self._config.nof_samples

        return slice(self._subbands_to_plot[0], self._subbands_to_plot[-1] + 1), \
               slice(nof_samples), \
               slice(self._antennas_to_plot[0], self._antennas_to_plot[-1] + 1)

    def initialise_plot(self):
        """ Initialise plot """
        plt.title("Anetnna Plot")
        self._figure.set_tight_layout(0.9)
        logging.info("AntennaPlotter: Initialised")

    def refresh_plot(self, input_data, obs_info):
        """ Update plot with new data
        :param input_data: Input data to plot
        :param obs_info: Observation information """

        # Loop over all beams to plot
        self._figure.clf()
        plt.title("Antena plot")

        for index, antenna in enumerate(self._antennas_to_plot):
            plt.plot(np.absolute(input_data[0, :, index]), label="Antenna %d" % antenna)
            plt.xlabel("Time")
            plt.ylabel("Power")
            plt.xlim([0, input_data.shape[1]])

        plt.legend()
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
        plt.show(block=False)

        logging.info("AntennaPlotter: Updated plot")
