import math
from matplotlib import pyplot as plt

import logging
import numpy as np

from pybirales.base.definitions import PipelineError
from pybirales.base.plotter import Plotter
from pybirales.blobs.dummy_data import DummyBlob
from pybirales.blobs.receiver_data import ReceiverBlob


class RawDataPlotter(Plotter):
    def __init__(self, config, input_blob):

        # Make sure that the input data blob is what we're expecting
        if type(input_blob) not in [DummyBlob, ReceiverBlob]:
            raise PipelineError("RawDataPlotter: Invalid input data type, should be DummyBlob or ReceiverBlob")

        # Call superclass initialiser
        super(RawDataPlotter, self).__init__(config, input_blob)

        # Get dimensions from input data blob
        input_shape = dict(self._input.shape)
        self._nants = input_shape['nants']
        self._nsamp = input_shape['nsamp']

        # Figure and axes placeholders
        self._antennas_to_plot = None
        self._figure = None
        self._axes = None

    def create_index(self):
        """ Create data index to get from blob """

        logging.info("RawDataPlotter: Called create index")

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

        # Initialise figure and turn on interactive plotting
        self._figure = plt.figure()
        plt.title("Antenna Plot")

        # # Add a subplot per antenna
        # self._axes = []
        #
        # if len(self._antennas_to_plot) == 2:
        #     axes = plt.subplot2grid((1, 2), (0, 0))
        #     plt.title("Antenna %d" % self._antennas_to_plot[0])
        #     self._axes.append(axes)
        #     axes = plt.subplot2grid((1, 2), (0, 1))
        #     plt.title("Antenna %d" % self._antennas_to_plot[1])
        #     self._axes.append(axes)
        # else:
        #     grid_dim_x = int(math.ceil(math.sqrt(len(self._antennas_to_plot))))
        #     grid_dim_y = int(math.ceil(math.sqrt(len(self._antennas_to_plot))))
        #
        #     for index, i in enumerate(self._antennas_to_plot):
        #         axes = plt.subplot2grid((grid_dim_x, grid_dim_y),
        #                                 (int((index / grid_dim_x)),
        #                                  int(index % grid_dim_x)))
        #         plt.title("Antenna %d" % i)
        #         self._axes.append(axes)

        # Tight layout
        self._figure.set_tight_layout(0.9)

        logging.info("RawDataPlotter: Called initialise plotter")
        pass

    def update_plot(self, input_data, obs_info):

        plt.cla()
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

        logging.info("RawDataPlotter: Called update plot")
        pass
