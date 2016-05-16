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
        self._figure = None
        self._axes = None

    def create_index(self):
        """ Create data index to get from blob """

        logging.info("RawDataPlotter: Called create index")

        # Check whether config file has any indexing defined
        antenna_range = None
        nof_samples = None
        if self._config is not None:
            if 'antenna_range' in self._config.settings():
                antenna_range = self._config.antenna_range
            if 'nof_samples' in self._config.settings():
                nof_samples = self._config.nof_samples

        return slice(None), slice(nof_samples), slice(antenna_range[0], antenna_range[1] + 1)

    def initialise_plot(self):

        # Initialise figure and turn on interactive plotting
        self._figure = plt.figure()

        logging.info("RawDataPlotter: Called initialise plotter")
        pass

    def update_plot(self, input_data, obs_info):
        plt.cla()
        plt.plot(np.arange(512))
        plt.title("Raw Antenna Plot")
        plt.show(block=False)
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()

        logging.info("RawDataPlotter: Called update plot")
        pass
