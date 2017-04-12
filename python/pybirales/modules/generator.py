import logging
import numpy as np
import time
from time import sleep
import datetime

from pybirales.base.definitions import PipelineError, ObservationInfo
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.dummy_data import DummyBlob
from scipy.signal import chirp
from pybirales.base import settings


# from pybirales.plotters.spectrogram_plotter import plotter


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

    def generate_corrdata(self):

        return np.ones((1, 1, 32, 70000), dtype=np.complex64)

    def process(self, obs_info, input_data, output_data):
        # Sampling rate
        frame_rate = 10e3
        n = self._nsamp

        start = n / frame_rate * self._counter

        ts = start + np.arange(n) / frame_rate
        f1 = 446.02233885
        f2 = 200.60375975
        noise_power = 0.1 * frame_rate / 2

        # Doppler shifted signal (from f1 to f2)
        freq = np.linspace(f1, f2, len(ts))
        ys = 100 * np.sin(2 * np.pi * freq * ts)
        ys += 10 * np.random.normal(scale=np.sqrt(noise_power), size=ts.shape)

        for i in range(self._nants):
            output_data[:, :, :, i] = ys
            # output_data[:, :, :, i] = 10*np.sin(np.arange(self._nsamp) * 0.5)
            # plotter.scatter(ys, ts, 'antenna_6_signal', i == 6)

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
