import logging
import numpy as np

from matplotlib import pyplot as plt

from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.plotter import Plotter
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob


class BandpassPlotter(Plotter):
    def __init__(self, config, input_blob, figure):

        # Make sure that the input data blob is what we're expecting
        if type(input_blob) is not ChannelisedBlob:
            raise PipelineError("ChannelisedDataPlotter: Invalid input data type, should be ChannelisedBlob")

        # Call superclass initialiser
        super(BandpassPlotter, self).__init__(config, input_blob, figure)

        # Get dimensions from input data blob
        input_shape = dict(self._input.shape)
        self._nbeams = input_shape['nbeams']
        self._nchans = input_shape['nchans']
        self._nsamp = input_shape['nsamp']

        # Figure and axes placeholders
        self._channels_to_plot = None
        self._beams_to_plot = None

    def create_index(self):
        """ Create data index to get from blob """

        # Check whether config file has any indexing defined
        nof_samples = None
        self._beams_to_plot = range(self._nbeams)
        self._channels_to_plot = range(self._nchans)

        if self._config is not None:
            if 'channel_range' in self._config.settings():
                if type(self._config.channel_range) is list:
                    self._channels_to_plot = range(self._config.channel_range[0], self._config.channel_range[1] + 1)
                else:
                    self._channels_to_plot = range(self._config.channel_range, self._config.channel_range + 1)

            if 'beam_range' in self._config.settings():
                if type(self._config.beam_range) is list:
                    self._beams_to_plot = range(self._config.beam_range[0], self._config.beam_range[1] + 1)
                else:
                    self._beams_to_plot = range(self._config.beam_range, self._config.beam_range + 1)

            if 'nof_samples' in self._config.settings():
                nof_samples = self._config.nof_samples

        polarization = 0
        beams_range = slice(self._beams_to_plot[0], self._beams_to_plot[-1] + 1)
        channels_range = slice(self._channels_to_plot[0], self._channels_to_plot[-1] + 1)
        samples_range = slice(nof_samples)

        return polarization, beams_range, channels_range, samples_range

    def initialise_plot(self):
        """ Initialise plot """
        plt.title("BandpassPlot")
        self._figure.set_tight_layout(0.9)
        logging.info("BandpassPlotter: Initialised")

    def refresh_plot(self, input_data, obs_info):
        """ Update plot with new data
        :param input_data: Input data to plot
        :param obs_info: Observation information """

        # Check if data is valid
        if 'nants' not in obs_info:
            return

        # Loop over all beams to plot
        self._figure.clf()
        plt.title("Bandpass plot")

        for index, beam in enumerate(self._beams_to_plot):
            # input_data[index, self._nchans / 2 - 1, :] = input_data[index, self._nchans / 2, :]
            values = np.sum(np.abs(input_data[index, :, :]), axis=1)
            plt.plot(values, label="Beam %d" % beam)
            plt.xlabel("Frequency")
            plt.ylabel("Power")
            plt.xlim([self._channels_to_plot[0], self._channels_to_plot[-1]])

        plt.legend()
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
        plt.show(block=False)
