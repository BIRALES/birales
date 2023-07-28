import logging
import time

import numpy as np
import datetime

from pybirales.pipeline.base.definitions import PipelineError, ObservationInfo
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.dummy_data import DummyBlob, GPUDummyBlob
from pybirales import settings

cu = None
try:
    import cupy as cu
except ImportError:
    pass


class DummyDataGenerator(ProcessingModule):
    """ Dummy data generator """

    def __init__(self, config, input_blob=None):

        # This module does not need an input_blob
        self._validate_data_blob(input_blob, valid_blobs=[type(None)])

        # Sanity checks on configuration
        if {'nof_antennas', 'nof_samples', 'nof_subbands'} - set(config.settings()) != set():
            raise PipelineError("DummyDataGenerator: Missing keys on configuration (nof_antennas, nof_samples, nsub)")
        self._nof_antennas = config.nof_antennas
        self._nof_samples = config.nof_samples
        self._nof_subbands = config.nof_subbands
        self._datatype = np.complex64

        # If GPUs need to be used, check whether CuPy is available
        if settings.manager.use_gpu and cu is None:
            logging.critical("GPU enabled but could not import CuPy.")
            raise PipelineError("CuPy could not be imported")

        # Call superclass initializer
        super(DummyDataGenerator, self).__init__(config, input_blob)

        self._counter = 1

        # Processing module name
        self.name = "Generator"

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """

        # Define data blob shape
        data_shape = [('nof_polarisations', 1), ('nof_subbands', self._nof_subbands), ('nof_samples', self._nof_samples), ('nof_antennas', self._nof_antennas)]

        # Create data shape on GPU or CPU
        if settings.manager.use_gpu:
            return GPUDummyBlob(data_shape, datatype=self._datatype, device=settings.manager.gpu_device_id)
        else:
            return DummyBlob(data_shape, datatype=self._datatype)

    def process(self, obs_info, input_data, output_data):
        """ Generate an array of ones with the input size """

        time.sleep(1)

        # Generate a sinusoidal signal
        signal = np.arange(self._nof_samples) * (1000.0 / settings.observation.samples_per_second) * 2 * np.pi
        signal = 50 * (np.cos(signal) + 1j * np.sin(signal))

        # The generator can generate the data on either the GPU or the host
        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id):
                output_data[:] = cu.asarray(signal[np.newaxis, np.newaxis, :, np.newaxis])
        else:
            output_data[:] = signal[np.newaxis, np.newaxis, :, np.newaxis]

        self._counter += 1

        # Create observation information
        obs_info = ObservationInfo()
        obs_info['sampling_time'] = 1. / settings.observation.samples_per_second
        obs_info['timestamp'] = datetime.datetime.utcnow()
        obs_info['nof_subbands'] = self._nof_subbands
        obs_info['nof_samples'] = self._nof_samples
        obs_info['nof_antennas'] = self._nof_antennas
        obs_info['nof_polarisations'] = 1
        obs_info['start_center_frequency'] = settings.observation.start_center_frequency
        obs_info['channel_bandwidth'] = settings.observation.channel_bandwidth

        logging.debug("Output data: %s shape: %s", np.sum(output_data), output_data.shape)

        return obs_info
