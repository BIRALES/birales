from math import ceil
import logging

import numba
import numpy as np
from numba import cuda

from pybirales import settings
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


@cuda.jit('void(complex64[:,:,:,:], complex64[:,:,:,:], int64[:,:], int32, int32, int32, int32, int32)')
def correlate_gpu(input_data, output_data, baseline_mapping, channel_start, channel_end, nants, integrations, nsamp):
    """ GPU correlation kernel """

    # Compute baseline number and check if it is out of bounds
    baseline = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if baseline >= nants * (nants - 1) // 2:
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
    samples_per_integration = nsamp // integrations

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
def correlate(input_data, output_data, channel_start, channel_end, nants, integrations, nsamp):
    sc = 0
    for c in range(channel_start, channel_end):
        baseline = 0
        for antenna1 in range(nants):
            for antenna2 in range(antenna1 + 1, nants):
                for i in range(nsamp // integrations):
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
        self._after_channelizer = True if type(input_blob) in [ChannelisedBlob, GPUChannelisedBlob] else False

        # Sanity checks on configuration
        if {'integrations'} - set(config.settings()) != set():
            raise PipelineError("Correlator: Missing keys on configuration. (integrations)")

        # Call superclass initialiser
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
        self._nbaselines = None
        self._nstokes = None
        self._nsamp = None
        self._nchans = None
        self._nants = None
        self._npols = None

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype

        baselines = int(0.5 * ((input_shape['nants'] ** 2) - input_shape['nants']))
        nstokes = input_shape['npols'] ** 2

        # Check if integration time is a multiple of nsamp
        if input_shape['nsamp'] % self._config.integrations != 0:
            raise PipelineError("Correlator: integration time must be a multiple of nsamp")

        # Generate output blob
        blob_type = GPUCorrelatedBlob if settings.manager.use_gpu else CorrelatedBlob
        nchans = input_shape['nchans'] if self._after_channelizer else input_shape['nsubs']
        if self._after_channelizer:
            blob_type = GPUChannelisedBlob if settings.manager.use_gpu else ChannelisedBlob

            if hasattr(settings.correlator, 'channel_start') and hasattr(settings.correlator, 'channel_end'):
                nchans = settings.correlator.channel_end - settings.correlator.channel_start

        # Define output blob metadata
        meta_data = [('nsamp', input_shape['nsamp'] / self._config.integrations),
                     ('nchans', nchans),
                     ('baselines', baselines),
                     ('stokes', nstokes)]

        # Generate output data blob
        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id):
                return blob_type(meta_data, datatype=datatype, device=settings.manager.gpu_device_id)
        else:
            return blob_type(meta_data, datatype=datatype)

    def process(self, obs_info, input_data, output_data):
        """ Perform correlation """

        # Update parameters
        self._nsamp = obs_info['nsamp']
        self._nchans = obs_info['nchans'] if self._after_channelizer else obs_info['nsubs']
        self._nants = obs_info['nants']
        self._npols = obs_info['npols']

        # Initialise variables during when processing first blob
        if self._iter_count == 1:
            self._nbaselines = int(0.5 * ((obs_info['nants'] ** 2) - obs_info['nants']))
            self._nstokes = obs_info['npols'] ** 2
            if hasattr(settings.correlator, 'channel_start') and hasattr(settings.correlator, 'channel_end'):
                self._channel_start = settings.correlator.channel_start
                self._channel_end = settings.correlator.channel_end
            else:
                self._channel_start = 0
                self._channel_end = self._nchans

            # Generate baseline mapping
            for i in range(obs_info['nants']):
                for j in range(i + 1, obs_info['nants']):
                    self._baseline_mapping.append([i, j])

            # Copy to GPU if using
            if settings.manager.use_gpu:
                with cu.cuda.Device(settings.manager.gpu_device_id):
                    self._baseline_mapping_gpu = cu.asarray(self._baseline_mapping)

        # Integration check
        if self._nsamp % self._integrations != 0:
            logging.warning("Number of integration not factor of number of samples, skipping buffer")
            return

        # GPU correlation
        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id):
                if self._after_channelizer:
                    self._current_input = cu.transpose(input_data, (0, 2, 1, 3))
                else:
                    self._current_input = cu.transpose(input_data, (0, 1, 3, 2))

                nof_threads = 32
                grid = (ceil(self._nbaselines / nof_threads), int(self._nsamp // self._integrations), self._nchans)
                correlate_gpu[grid, nof_threads](self._current_input, output_data,  self._baseline_mapping_gpu,
                                                 self._channel_start, self._channel_end, self._nants,
                                                 int(self._nsamp // self._integrations), self._nsamp)

        # CPU correlation
        else:
            if self._after_channelizer:
                self._current_input = np.transpose(input_data, (0, 2, 1, 3)).copy()
            else:
                self._current_input = np.transpose(input_data, (0, 1, 3, 2)).copy()

            correlate(self._current_input, output_data, self._channel_start, self._channel_end,
                      self._nants, self._integrations, self._nsamp)

        obs_info['nsamp'] = self._nsamp // self._integrations
        obs_info['sampling_time'] *= self._integrations
        obs_info['nbaselines'] = self._nbaselines
        obs_info['nstokes'] = self._nstokes

        return obs_info
