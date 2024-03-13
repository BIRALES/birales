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

        # If number of iterations is specified in configuration, load it
        self._nof_iterations = -1
        if 'nof_iterations' in config.settings():
            self._nof_iterations = config.nof_iterations

        # If GPUs need to be used, check whether CuPy is available
        if settings.manager.use_gpu and cu is None:
            logging.critical("GPU enabled but could not import CuPy.")
            raise PipelineError("CuPy could not be imported")

        # Define generation start time
        self._start_time = datetime.datetime.utcnow()

        # Call superclass initializer
        super(DummyDataGenerator, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Generator"

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """

        # Define data blob shape
        data_shape = [('nof_polarisations', 1), ('nof_subbands', self._nof_subbands),
                      ('nof_samples', self._nof_samples), ('nof_antennas', self._nof_antennas)]

        # Create data shape on GPU or CPU
        if settings.manager.use_gpu:
            return GPUDummyBlob(data_shape, datatype=self._datatype, device=settings.manager.gpu_device_id)
        else:
            return DummyBlob(data_shape, datatype=self._datatype)

    def process(self, obs_info, input_data, output_data):
        """ Generate an array of ones with the input size """

        # Check number of iterations, and signal stop pipeline if this is the last one
        if self._nof_iterations != -1 and self._iteration_counter > self._nof_iterations:
            obs_info['stop_pipeline_at'] = self._iteration_counter
            self.stop()
            return obs_info

        # Generate a sinusoidal signal
        signal = np.arange(self._nof_samples) * (1000.0 / settings.observation.samples_per_second) * 2 * np.pi
        signal = 50 * (np.cos(signal) + 1j * np.sin(signal))

        # The generator can generate the data on either the GPU or the host
        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id):
                output_data[:] = cu.asarray(signal[np.newaxis, np.newaxis, :, np.newaxis])
        else:
            output_data[:] = signal[np.newaxis, np.newaxis, :, np.newaxis]

        # Update timestamp
        sampling_time = 1. / settings.observation.samples_per_second
        timestamp = (self._start_time +
                     datetime.timedelta(seconds=self._iteration_counter * self._nof_samples * sampling_time))

        # Create observation information
        obs_info = ObservationInfo()
        obs_info['sampling_time'] = sampling_time
        obs_info['timestamp'] = timestamp
        obs_info['nof_subbands'] = self._nof_subbands
        obs_info['nof_samples'] = self._nof_samples
        obs_info['nof_antennas'] = self._nof_antennas
        obs_info['nof_polarisations'] = 1
        obs_info['start_center_frequency'] = settings.observation.start_center_frequency
        obs_info['channel_bandwidth'] = settings.observation.channel_bandwidth
        obs_info['observation_name'] = settings.observation.name
        obs_info['transmitter_frequency'] = settings.observation.transmitter_frequency
        obs_info['declinations'] = settings.beamformer.reference_declinations

        logging.debug("Output data: %s shape: %s", np.sum(output_data), output_data.shape)

        return obs_info