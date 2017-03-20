import logging
import numpy as np
import time

from pybirales.base.definitions import PipelineError, ObservationInfo
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.dummy_data import DummyBlob
from scipy.signal import chirp
from pybirales.plotters.spectrogram_plotter import plotter


class DummyDataGenerator(ProcessingModule):
    """ Dummy data generator """

    def __init__(self, config, input_blob=None):

        # This module does not need an input_blob
        if input_blob is not None:
            raise PipelineError("DummyDataGenerator: Invalid input data type, should be None")

        # Sanity checks on configuration
        if {'nants', 'nsamp', 'nsubs', 'npols', 'nbits', 'complex'} - set(config.settings()) != set():
            raise PipelineError("DummyDataGenerator: Missing keys on configuration "
                                "(nants, nsamp, nsub, 'npols', 'nbits', complex")
        self._nants = config.nants
        self._nsamp = config.nsamp
        self._nsubs = config.nsubs
        self._npols = config.npols
        self._nbits = config.nbits
        self._complex = config.complex

        # Define data type
        if self._nbits == 64 and self._complex:
            self._datatype = np.complex64
        else:
            raise PipelineError("DummyDataGenerator: Unsupported data type (bits, complex)")

        # Call superclass initialiser
        super(DummyDataGenerator, self).__init__(config, input_blob)

        self._counter = 1

        # Processing module name
        self.name = "Generator"

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return DummyBlob(self._config, [('npols', self._npols),
                                        ('nsubs', self._nsubs),
                                        ('nsamp', self._nsamp),
                                        ('nants', self._nants)],
                         datatype=self._datatype)

    def process(self, obs_info, input_data, output_data):
        # time.sleep(0.05)

        # Perform operations
        # output_data[:] = np.ones((self._nsubs, self._nsamp, self._nants), dtype=self._datatype)
        # self._counter += 1

        for i in range(self._nants):
            # output_data[:, :, :, i] = np.random.rand() * np.sin(2**np.logspace(0.0001, 4, num=self._nsamp, base=2))
            # output_data[:, :, :, i] = np.sin(2**np.logspace(0.001, 4, num=self._nsamp, base=2))
            samples = np.linspace(0, self._nsamp, num=self._nsamp)

            d1 = chirp(samples,
                       f0=410.00102776367186,
                       f1=410.00082749206540,
                       t1=1, method='linear')
            #
            # d2 = 1j * chirp(samples,
            #                 f0=410.00102776367186,
            #                 f1=410.00082749206540,
            #                 t1=1, method='linear')

            noise = np.random.rand(self._nsamp)

            # d2 = 10 * np.random.rand() * chirp(samples,
            #                                    f0=390e6,
            #                                    f1=400e6,
            #                                    t1=1, method='linear')
            d = d1
            output_data[:, :, :, i] = d

            plotter.scatter(d + noise, samples, 'antenna_6_signal', i == 6)

        # output_data[:] = np.ones((self._npols, self._nsubs, self._nsamp, self._nants), dtype=self._datatype)

        self._counter += 1

        # Create observation information
        obs_info = ObservationInfo()
        obs_info['sampling_time'] = 0.0
        obs_info['timestamp'] = 0.0
        obs_info['nsubs'] = self._nsubs
        obs_info['nsamp'] = self._nsamp
        obs_info['nants'] = self._nants
        obs_info['npols'] = self._npols
        obs_info['channel_bandwidth'] = 0.1
        obs_info['start_center_frequency'] = 400.

        logging.info("Generated data")
        logging.debug("Input data: %s", np.sum(input_data))
        logging.debug("Output data: %s shape: %s", np.sum(output_data), output_data.shape)
        return obs_info
