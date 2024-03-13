from math import ceil
import logging

import numba
import numpy as np
from numba import cuda

from pybirales import settings
from pybirales.pipeline.base.cuda_wrapper import try_cuda_jit
from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob, GPUChannelisedBlob
from pybirales.pipeline.blobs.correlated_data import CorrelatedBlob, GPUCorrelatedBlob
from pybirales.pipeline.blobs.dummy_data import DummyBlob, GPUDummyBlob
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob, GPUReceiverBlob

cu = None
try:
    import cupy as cu
    from cupyx.scipy.fft import fft, fftshift
except ImportError:
    pass


@try_cuda_jit('void(complex64[:,:,:,:], complex64[:,:,:,:], int64[:,:], int32, int32, int32, int32, int32)')
def correlate_gpu(input_data, output_data, baseline_mapping, channel_start, channel_end, nof_antennas, integrations, nof_samples):
    """ GPU correlation kernel """

    # Compute baseline number and check if it is out of bounds
    baseline = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if baseline >= nof_antennas * (nof_antennas - 1) // 2:
        return

    # Integrations are processed along grid Y direction
    integration = cuda.blockIdx.y
    if integration >= integrations:
        return

    # Channels processing along grid Z direction
    channel = channel_start + cuda.blockIdx.z
    if channel >= channel_end:
        return

    # Compute samples per integration
    samples_per_integration = nof_samples // integrations

    # Get thread's antenna pair
    ant_x, ant_y = baseline_mapping[baseline]

    # Accumulate across samples for thread's baseline
    accumulator = 0+0j
    for sample in range(samples_per_integration):
        ant_1_voltage = input_data[0, channel, ant_x, integration * samples_per_integration + sample]
        ant_2_voltage = input_data[0, channel, ant_y, integration * samples_per_integration + sample]
        ant_2_voltage = ant_2_voltage.real - 1j * ant_2_voltage.imag
        accumulator += ant_1_voltage * ant_2_voltage

    # Save result to memory
    output_data[integration, channel, baseline, 0] = accumulator


@numba.jit(nopython=True, nogil=True)
def correlate(input_data, output_data, channel_start, channel_end, nof_antennas, integrations, nof_samples):
    sc = 0
    for c in range(channel_start, channel_end):
        baseline = 0
        for antenna1 in range(nof_antennas):
            for antenna2 in range(antenna1 + 1, nof_antennas):
                for i in range(nof_samples // integrations):
                    output_data[i, sc, baseline, 0] = np.dot(
                        input_data[0, c, antenna1, i * integrations:(i + 1) * integrations],
                        np.conj(input_data[0, c, antenna2, i * integrations:(i + 1) * integrations]))
                baseline += 1

        sc += 1


class Correlator(ProcessingModule):
    """ Correlator module """

    def __init__(self, config, input_blob=None):

        self._validate_data_blob(input_blob, valid_blobs=[ReceiverBlob, GPUReceiverBlob,
                                                          DummyBlob, GPUDummyBlob,
                                                          ChannelisedBlob, GPUChannelisedBlob])

        # Check if we're dealing with channelised data or receiver data
        self._after_channeliser = True if type(input_blob) in [ChannelisedBlob, GPUChannelisedBlob] else False

        # Sanity checks on configuration
        if {'integrations'} - set(config.settings()) != set():
            raise PipelineError("Correlator: Missing keys on configuration. (integrations)")

        # Call superclass initializer
        super(Correlator, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Correlator"

        # Populate variables
        self._integrations = config.integrations

        # Define parameters
        self._baseline_mapping_gpu = None
        self._baseline_mapping = []
        self._current_input = None
        self._current_output = None
        self._channel_start = None
        self._channel_end = None
        self._nof_baselines = None
        self._nof_stokes = None
        self._nof_samples = None
        self._nof_channels = None
        self._nof_antennas = None
        self._nof_polarisations = None

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype

        baselines = int(0.5 * ((input_shape['nof_antennas'] ** 2) - input_shape['nof_antennas']))
        nof_stokes = input_shape['nof_polarisations'] ** 2

        # Check if integration time is a multiple of nof_samples
        if input_shape['nof_samples'] % self._config.integrations != 0:
            raise PipelineError("Correlator: integration time must be a multiple of nof_samples")

        # Generate output blob
        blob_type = GPUCorrelatedBlob if settings.manager.use_gpu else CorrelatedBlob
        nof_channels = input_shape['nof_channels'] if self._after_channeliser else input_shape['nof_subbands']
        if self._after_channeliser:
            blob_type = GPUChannelisedBlob if settings.manager.use_gpu else ChannelisedBlob

            if hasattr(settings.correlator, 'channel_start') and hasattr(settings.correlator, 'channel_end'):
                nof_channels = settings.correlator.channel_end - settings.correlator.channel_start

        # Define output blob metadata
        meta_data = [('nof_samples', input_shape['nof_samples'] / self._config.integrations),
                     ('nof_channels', nof_channels),
                     ('baselines', baselines),
                     ('stokes', nof_stokes)]

        # Generate output data blob
        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id):
                return blob_type(meta_data, datatype=datatype, device=settings.manager.gpu_device_id)
        else:
            return blob_type(meta_data, datatype=datatype)

    def process(self, obs_info, input_data, output_data):
        """ Perform correlation """

        # Update parameters
        self._nof_samples = obs_info['nof_samples']
        self._nof_channels = obs_info['nof_channels'] if self._after_channeliser else obs_info['nof_subbands']
        self._nof_antennas = obs_info['nof_antennas']
        self._nof_polarisations = obs_info['nof_polarisations']

        # Initialise variables when processing first blob
        if self._iter_count == 1:
            self._nof_baselines = int(0.5 * ((obs_info['nof_antennas'] ** 2) - obs_info['nof_antennas']))
            self._nof_stokes = obs_info['nof_polarisations'] ** 2
            if hasattr(settings.correlator, 'channel_start') and hasattr(settings.correlator, 'channel_end'):
                self._channel_start = settings.correlator.channel_start
                self._channel_end = settings.correlator.channel_end
            else:
                self._channel_start = 0
                self._channel_end = self._nof_channels

            # Generate baseline mapping
            for i in range(obs_info['nof_antennas']):
                for j in range(i + 1, obs_info['nof_antennas']):
                    self._baseline_mapping.append([i, j])

            # Copy to GPU if using
            if settings.manager.use_gpu:
                with cu.cuda.Device(settings.manager.gpu_device_id):
                    self._baseline_mapping_gpu = cu.asarray(self._baseline_mapping)

        # Integration check
        if self._nof_samples % self._integrations != 0:
            logging.warning("Number of integration not factor of number of samples, skipping buffer")
            return

        # GPU correlation
        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id):
                if self._after_channeliser:
                    self._current_input = cu.transpose(input_data, (0, 2, 1, 3))
                else:
                    self._current_input = cu.transpose(input_data, (0, 1, 3, 2))

                nof_threads = 32
                grid = (ceil(self._nof_baselines / nof_threads), int(self._nof_samples // self._integrations), self._nof_channels)
                correlate_gpu[grid, nof_threads](self._current_input, output_data,  self._baseline_mapping_gpu,
                                                 self._channel_start, self._channel_end, self._nof_antennas,
                                                 int(self._nof_samples // self._integrations), self._nof_samples)

        # CPU correlation
        else:
            if self._after_channeliser:
                self._current_input = np.transpose(input_data, (0, 2, 1, 3)).copy()
            else:
                self._current_input = np.transpose(input_data, (0, 1, 3, 2)).copy()

            correlate(self._current_input, output_data, self._channel_start, self._channel_end,
                      self._nof_antennas, self._integrations, self._nof_samples)

        obs_info['nof_samples'] = self._nof_samples // self._integrations
        obs_info['sampling_time'] *= self._integrations
        obs_info['nof_baselines'] = self._nof_baselines
        obs_info['nof_stokes'] = self._nof_stokes

        return obs_info
