import logging
import numpy as np

from matplotlib import pyplot as plt

from pybirales.base.definitions import PipelineError
from pybirales.base.plotter import Plotter
from pybirales.blobs.beamformed_data import BeamformedBlob


class BeamformedDataPlotter(Plotter):
    def __init__(self, config, input_blob, figure):

        # Make sure that the input data blob is what we're expecting
        if type(input_blob) is not BeamformedBlob:
            raise PipelineError("BeamformedDataPlotter: Invalid input data type, should be BeamformedBlob")

        # Call superclass initialiser
        super(BeamformedDataPlotter, self).__init__(config, input_blob, figure)

        # Get dimensions from input data blob
        input_shape = dict(self._input.shape)
        self._nbeams = input_shape['nbeams']
        self._nsamp = input_shape['nsamp']

        # Figure and axes placeholders
        self._beams_to_plot = None
        self._axes = None

    def create_index(self):
        """ Create data index to get from blob """

        # Check whether config file has any indexing defined range(self._config.beam_range, self._config.beam_range + 1)
        nof_samples = None
        self._beams_to_plot = range(self._nbeams)

        # TODO: This only plots the first subband in the data, update to be able to select multiple subbands
        if self._config is not None:
            if 'beam_range' in self._config.settings():
                if type(self._config.beam_range) is list:
                    self._beams_to_plot = range(self._config.beam_range[0], self._config.beam_range[1] + 1)
                else:
                    print(range(self._config.beam_range, self._config.beam_range + 1))
                    self._beams_to_plot = range(self._config.beam_range, self._config.beam_range + 1)

            if 'nof_samples' in self._config.settings():
                nof_samples = self._config.nof_samples

        return slice(0, 1, None), slice(self._beams_to_plot[0], self._beams_to_plot[-1] + 1), slice(0, 1), slice(nof_samples)

    def initialise_plot(self):
        """ Initialise plot """

        # Initialise figure
        plt.title("Beamformer Plot")
        self._figure.set_tight_layout(0.9)

    def refresh_plot(self, input_data, obs_info):
        """ Update plot with new data
        :param input_data: Input data to plot
        :param obs_info: Observation information """

        self._figure.clf()
        plt.title("Beamformer plot")
        for index, beam in enumerate(self._beams_to_plot):
            plt.plot(np.abs(input_data[0, index, 0, :]), label="Beam %d" % beam)
            plt.xlabel("Time")
            plt.ylabel("Power")
            plt.xlim([0, input_data.shape[3]])

        plt.legend()
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
        plt.show(block=False)
