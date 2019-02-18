import logging
import numpy as np
import datetime

from pybirales.pipeline.base.definitions import PipelineError, ObservationInfo
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.dummy_data import DummyBlob
from pybirales import settings


class DummyDataGenerator(ProcessingModule):
    """ Dummy data generator """

    def __init__(self, config, input_blob=None):

        # This module does not need an input_blob
        self._validate_data_blob(input_blob, valid_blobs=[type(None)])

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

    def generate_corrdata(self):

        return np.ones((1, 1, 32, 70000), dtype=np.complex64)

    def process(self, obs_info, input_data, output_data):
        # Sampling rate
        frame_rate = 10e3
        n = self._nsamp

        start = n / frame_rate * 1

        # ts = start + np.arange(n) / frame_rate
        # noise_power = 0.1 * frame_rate / 2
        #
        # for i in range(self._nants):
        #     output_data[:, :, :, i] = np.random.rand() * np.sin(2 ** np.logspace(0.0001, 4, num=self._nsamp, base=2))

        output_data[:] = np.ones((self._nsubs, self._nsamp, self._nants), dtype=self._datatype)

        # f1 = 447.02233885
        # f2 = 200.605
        # for i in range(self._nants):
        #     # f1 = np.random.uniform(445.02233885, 447.02233885)
        #     # f2 = np.random.uniform(200.60375975, 201.60375975)
        #
        #     # Doppler shifted signal (from f1 to f2)
        #     freq = np.linspace(f1, f2, len(ts))
        #     ys = np.sin(2 * np.pi * freq * ts)
        #
        #     ys += 0.01 * np.random.normal(scale=np.sqrt(noise_power), size=ts.shape)
        #
        #     output_data[:, :, :, i] = ys
        #     # output_data[:, :, :, i] = 10*np.sin(np.arange(self._nsamp) * 0.5)
        #     # plotter.scatter(ys, ts, 'antenna_0_signal', True)

        self._counter += 1

        # output_data = self.generate_corrdata()
        # Create observation information
        obs_info = ObservationInfo()
        obs_info['sampling_time'] = 1. / settings.observation.samples_per_second
        obs_info['timestamp'] = datetime.datetime.utcnow()
        obs_info['nsubs'] = self._nsubs
        obs_info['nsamp'] = self._nsamp
        obs_info['nants'] = self._nants
        obs_info['npols'] = self._npols
        obs_info['channel_bandwidth'] = 0.1
        obs_info['start_center_frequency'] = 400.

        logging.debug("Input data: %s", np.sum(input_data))
        logging.debug("Output data: %s shape: %s", np.sum(output_data), output_data.shape)

        return obs_info
