import logging
import math
import numpy as np
import logging as log
from matplotlib import pyplot as plt

from pybirales.base.definitions import PipelineError
from pybirales.base.plotter import Plotter
from pybirales.blobs.channelised_data import ChannelisedBlob


class ChannelisedDataPlotter(Plotter):
    def __init__(self, config, input_blob, figure):

        # Make sure that the input data blob is what we're expecting
        if type(input_blob) is not ChannelisedBlob:
            raise PipelineError("ChannelisedDataPlotter: Invalid input data type, should be ChannelisedBlob")

        # Call superclass initialiser
        super(ChannelisedDataPlotter, self).__init__(config, input_blob, figure)

        # Get dimensions from input data blob
        input_shape = dict(self._input.shape)
        self._nbeams = input_shape['nbeams']
        self._nchans = input_shape['nchans']
        self._nsamp = input_shape['nsamp']

        # Figure and axes placeholders
        self._channels_to_plot = None
        self._beams_to_plot = None
        self._axes = None

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
                    self._channels_to_plot = range(self._config.channel_range, self._config.channel_range)

            if 'beam_range' in self._config.settings():
                if type(self._config.beam_range) is list:
                    self._beams_to_plot = range(self._config.beam_range[0], self._config.beam_range[1])
                else:
                    self._beams_to_plot = range(self._config.beam_range, self._config.beam_range + 1)

            if 'nof_samples' in self._config.settings():
                nof_samples = self._config.nof_samples

        polarization = 0
        beams_range = slice(self._beams_to_plot[0], self._beams_to_plot[-1] + 1)
        channels_range = slice(self._channels_to_plot[0], self._channels_to_plot[-1] + 1)
        samples_range = slice(0, nof_samples)

        return polarization, beams_range, channels_range, samples_range

    def initialise_plot(self):
        """ Initialise plot """

        # Add a subplot per antenna
        self._axes = []

        if len(self._beams_to_plot) == 2:
            axes = plt.subplot2grid((1, 2), (0, 0))
            plt.title("Beam %d" % self._beams_to_plot[0])
            self._axes.append(axes)
            axes = plt.subplot2grid((1, 2), (0, 1))
            plt.title("Beam %d" % self._beams_to_plot[1])
            self._axes.append(axes)
        else:
            grid_dim_x = int(math.ceil(math.sqrt(len(self._beams_to_plot))))
            grid_dim_y = int(math.ceil(math.sqrt(len(self._beams_to_plot))))

            for index, i in enumerate(self._beams_to_plot):
                axes = plt.subplot2grid((grid_dim_x, grid_dim_y),
                                        (int((index / grid_dim_x)),
                                         int(index % grid_dim_x)))
                plt.title("Beam %d" % i)
                self._axes.append(axes)

    def refresh_plot(self, input_data, obs_info):
        """ Update plot with new data
        :param input_data: Input data to plot
        :param obs_info: Observation information """

        log.debug("Input data: %s shape: %s", np.sum(input_data), input_data.shape)
        if not input_data.any():
            log.warning('Nothing to plot. Input data is empty')
            return

        # Loop over all beams to plot
        im = None
        for index, beam in enumerate(self._beams_to_plot):
            self._axes[index].cla()
            input_data[index, int(self._nchans / 2) - 1, :] = input_data[index, int(self._nchans / 2), :]

            # input_data[index, 255, :] = input_data[index, 255, :]

            im = self._axes[index].imshow(np.abs(input_data[index, :, :]),
                                          aspect='auto', interpolation='none')
            self._axes[index].set_xlabel("Time")
            self._axes[index].set_ylabel("Channel")
            self._axes[index].set_title("Beam %d" % beam)

        cax = self._figure.add_axes([0.95, 0.1, 0.01, 0.8])
        self._figure.subplots_adjust(right=0.9)
        self._figure.colorbar(im, cax)
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
        plt.show(block=False)

        logging.info("ChannelisedDataPlotter: Updated plot")
