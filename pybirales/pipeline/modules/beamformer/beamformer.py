import logging
import math

import numba
import numpy as np
from numba import njit, prange, cuda
from pybirales import settings

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
def beamformer_python(nbeams, data, weights, output):
    for b in prange(nbeams):
        output[0, b, 0, :] = np.dot(data, weights[0, b, :])


@cuda.jit('void(complex64[:,:,:,:], complex64[:,:,:,:], complex64[:,:,:])', fastmath=True)
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
        if {'nbeams', 'pointings', 'reference_declination'} \
                - set(config.settings()) != set():
            raise PipelineError("Beamformer: Missing keys on configuration "
                                "(nbeams, nants, pointings)")

        self._nbeams = config.nbeams

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
        data_shape = [('npols', input_shape['npols']), ('nbeams', self._nbeams),
                      ('nsubs', input_shape['nsubs']), ('nsamp', input_shape['nsamp'])]

        if settings.manager.use_gpu:
            return GPUBeamformedBlob(data_shape, datatype=datatype, device=settings.manager.gpu_device_id)
        else:
            return BeamformedBlob(data_shape, datatype=datatype)

    def _initialise(self, nsubs, nants):
        """ Initialise pointing """

        # Create pointing instance
        if self._pointing is None:
            self._pointing = Pointing(self._config, nsubs, nants)
            if self._disable_antennas is not None:
                self._pointing.disable_antennas(self._disable_antennas)

    def process(self, obs_info, input_data, output_data):

        # Get data information
        nsamp = obs_info['nsamp']
        nsubs = obs_info['nsubs']
        nants = obs_info['nants']
        npols = obs_info['npols']

        # If pointing is not initialised then this is the first input blob that's being processed.
        # Initialise pointing and GPU arrays
        if self._pointing is None:
            self._initialise(nsubs, nants)

            # If using GPU, copy weights to GPU
            if settings.manager.use_gpu:
                with cu.cuda.Device(settings.manager.gpu_device_id):
                    self._weights_gpu = cu.asarray(self._pointing.weights)

        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id) as d:
                # TODO: If input blob is a CPU blob, copy to GPU
                # TODO: Handle case where output blob is not on GPU

                # Transpose input to improve memory access in beamformer
                input_data = cu.transpose(input_data, (0, 1, 3, 2))

                # Run beamforming kernel
                grid = (math.ceil(nsamp / 128), self._nbeams)
                block_size = 128
                beamformer_gpu[grid, block_size](input_data, output_data, self._weights_gpu)
                d.synchronize()
        else:
            # TODO: Extract pols and sub-bands properly
            beamformer_python(self._nbeams, input_data[0, 0], self._pointing.weights, output_data)

        # Update observation information
        obs_info['nbeams'] = self._nbeams
        obs_info['pointings'] = self._config.pointings
        obs_info['beam_az_el'] = self._pointing.beam_az_el
        obs_info['declination'] = self._pointing._reference_declination

        return obs_info
