import logging
import math

import numba
import numpy as np
from numba import njit, prange, cuda

from pybirales import settings
from pybirales.pipeline.base.cuda_wrapper import try_cuda_jit
from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.beamformed_data import BeamformedBlob, GPUBeamformedBlob
from pybirales.pipeline.blobs.dummy_data import DummyBlob, GPUDummyBlob
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob, GPUReceiverBlob
from pybirales.pipeline.modules.beamformer.pointing import Pointing

cu = None
try:
    import cupy as cu
except ImportError:
    pass

# Define the maximum number of antennas to generate shared memory buffer in GPU
MAX_ANTENNAS = 256


@njit(parallel=True, fastmath=True)
def beamformer_python(nof_beams, data, weights, output):
    for b in prange(nof_beams):
        output[0, b, 0, :] = np.dot(data, weights[0, b, :])


@try_cuda_jit('void(complex64[:,:,:,:], complex64[:,:,:,:], complex64[:,:,:])', fastmath=True)
def beamformer_gpu(input_data, output_data, weights):
    # Input in pol/sub/samp/ant order
    # Output in pol/beam/sub/samp order
    # Weights in sub/beam/ant order

    # Compute sample index and check whether thread is out of bounds
    sample = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    nof_antennas = input_data.shape[-2]
    beam = cuda.blockIdx.y

    # Thread only continues if it's associated spectrum is valid
    if sample >= input_data.shape[-1]:
        return

    # Use a shared memory block to store the weights of associated beam
    shared_memory = cuda.shared.array(MAX_ANTENNAS, numba.complex64)

    # Cooperative load of pointing coefficient for current beam
    for t in range(cuda.threadIdx.x, nof_antennas, cuda.blockDim.x):
        shared_memory[t] = weights[0, beam, t]
    cuda.syncthreads()

    # Each thread computes the beamformed value of a single time sample
    beamformed_value = 0 + 0j
    for a in range(input_data.shape[-2]):
        beamformed_value += input_data[0, 0, a, sample] * shared_memory[a]

    # Save result to output
    output_data[0, beam, 0, sample] = beamformed_value


class Beamformer(ProcessingModule):
    """ Beamformer processing module """

    def __init__(self, config, input_blob=None):

        # Check whether input blob type is compatible
        self._validate_data_blob(input_blob, valid_blobs=[DummyBlob, GPUDummyBlob, ReceiverBlob, GPUReceiverBlob])

        # Sanity checks on configuration
        if {'nof_subarrays', 'nof_beams_per_subarray'} \
                - set(config.settings()) != set():
            raise PipelineError("Beamformer: Missing keys on configuration "
                                "(nof_subarrays, nof_beams_per_subarray)")

        # This must be populated
        self._nof_beams = config.nof_subarrays * config.nof_beams_per_subarray

        # If GPUs need to be used, check whether CuPy is available
        if settings.manager.use_gpu and cu is None:
            logging.critical("GPU enabled but could not import CuPy.")
            raise PipelineError("CuPy could not be imported")

        self._disable_antennas = None
        if 'disable_antennas' in config.settings():
            self._disable_antennas = config.disable_antennas

        # Create placeholder for pointing class instance
        self._pointing = None
        self._weights_gpu = None

        # Call superclass initializer
        super(Beamformer, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Beamformer"

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype

        # Create output blob
        data_shape = [('nof_polarisations', input_shape['nof_polarisations']), ('nof_beams', self._nof_beams),
                      ('nof_subbands', input_shape['nof_subbands']), ('nof_samples', input_shape['nof_samples'])]

        if settings.manager.use_gpu:
            return GPUBeamformedBlob(data_shape, datatype=datatype, device=settings.manager.gpu_device_id)
        else:
            return BeamformedBlob(data_shape, datatype=datatype)

    def _initialise(self, nof_subbands, nof_antennas):
        """ Initialise pointing """

        # Create pointing instance
        if self._pointing is None:
            self._pointing = Pointing(self._config, nof_subbands, nof_antennas)
            if self._disable_antennas is not None:
                self._pointing.disable_antennas(self._disable_antennas)
            self._nof_beams = self._pointing.number_of_pointings

    def process(self, obs_info, input_data, output_data):

        # Get data information
        nof_samples = obs_info['nof_samples']
        nof_subbands = obs_info['nof_subbands']
        nof_antennas = obs_info['nof_antennas']

        # If pointing is not initialised then this is the first input blob that's being processed.
        # Initialise pointing and GPU arrays
        if self._pointing is None:
            self._initialise(nof_subbands, nof_antennas)

            # If using GPU, copy weights to GPU
            if settings.manager.use_gpu:
                with cu.cuda.Device(settings.manager.gpu_device_id):
                    self._weights_gpu = cu.asarray(self._pointing.pointing_weights)

        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id) as d:
                # TODO: If input blob is a CPU blob, copy to GPU
                # TODO: Handle case where output blob is not on GPU

                # Transpose input to improve memory access in beamformer
                input_data = cu.transpose(input_data, (0, 1, 3, 2))

                # Run beamforming kernel
                grid = (math.ceil(nof_samples / 128), self._nof_beams)
                block_size = 128
                beamformer_gpu[grid, block_size](input_data, output_data, self._weights_gpu)
                d.synchronize()
        else:
            # TODO: Extract pols and sub-bands properly
            beamformer_python(self._nof_beams, input_data[0, 0], self._pointing.pointing_weights, output_data)

        # Update observation information
        obs_info['nof_beams'] = self._nof_beams
        obs_info['pointings'] = self._pointing.pointings
        obs_info['beam_az_el'] = self._pointing.beam_azimuth_elevation
        obs_info['declinations'] = self._pointing.reference_declinations

        return obs_info
