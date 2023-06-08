import math
from datetime import timedelta

import cupy
import numba
import numpy as np
from numba import cuda
from pybirales import settings

from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.beamformed_data import BeamformedBlob, GPUBeamformedBlob
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob, GPUChannelisedBlob
from pybirales.pipeline.blobs.dummy_data import DummyBlob, GPUDummyBlob
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob, GPUReceiverBlob

cu = None
try:
    import cupy as cu
    from cupyx.scipy.fft import fft, fftshift
except ImportError:
    pass


@numba.jit(nopython=True, nogil=True)
def apply_fir_filter(data, fir_filter, output, ntaps, nchans):
    """
    Optimised filter function using numpy and numba
    :param data: Input data pointer
    :param fir_filter: Filter coefficients pointer
    :param output: Output data pointer
    :param ntaps: Number of taps
    :param nchans: Number of channels
    """
    nof_spectra = (len(data) - ntaps * nchans) // nchans
    for n in range(nof_spectra):
        temp = data[n * nchans: n * nchans + nchans * ntaps] * fir_filter
        for j in range(1, ntaps):
            temp[:nchans] += temp[j * nchans: (j + 1) * nchans]
        output[:, n] = temp[:nchans]


@cuda.jit('void(complex64[:,:,:,:], float64[:], complex64[:,:,:,:,:], int32, int32, int32)', fastmath=True)
def apply_fir_filter_cuda(input_data, fir_filter, output_data, nof_spectra, nchans, ntaps):
    """
    Optimised filter function using numpy and numba
    :param input_data: Input data pointer
    :param fir_filter: Filter coefficients pointer
    :param output_data: Output data pointer
    :param nof_spectra: Number of spectra to process
    :param ntaps: Number of taps
    :param nchans: Number of channels
    """

    # NOTE: This assumes that npols and nsubs are 1 for the time being
    spectrum = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    stream = cuda.blockIdx.y

    # Get relevant parts of input and output buffers
    input_ptr = input_data[0, stream, 0]
    output_ptr = output_data[0, stream, 0]

    if spectrum >= nof_spectra:
        return

    # Loop across channels and each tap multiplied by the associated filter value to generate
    # the filtered channel response
    for channel in range(nchans):
        temp_value = 0
        for tap in range(ntaps):
            temp_value += input_ptr[spectrum * nchans + tap * nchans + channel] * fir_filter[tap * nchans + channel]
        output_ptr[channel, spectrum] = temp_value


class PFB(ProcessingModule):
    """ PPF processing module """

    def __init__(self, config, input_blob=None):

        # Check whether input blob type is compatible
        if type(input_blob) not in [BeamformedBlob, GPUBeamformedBlob,
                                    DummyBlob, GPUDummyBlob,
                                    ReceiverBlob, GPUReceiverBlob]:
            raise PipelineError("PFB: Invalid input data type, should be BeamformedBlob, DummyBlob or ReceiverBlob")

        # Check if we're dealing with beamformed data or receiver data
        self._after_beamformer = True if type(input_blob) in [GPUBeamformedBlob, BeamformedBlob] else False

        # Sanity checks on configuration
        if {'nchans', 'ntaps'} - set(config.settings()) != set():
            raise PipelineError("PPF: Missing keys on configuration. (nchans, ntaps, nsamp, nbeams)")
        self._bin_width_scale = 1.0
        self._nchans = config.nchans
        self._ntaps = config.ntaps

        # Call superclass initialiser
        super(PFB, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Channeliser"

        # Variable below will be populated in generate_output_blob
        self._filter = None
        self._filter_gpu = None
        self._filtered = None
        self._temp_input = None
        self._nbeams = None
        self._nsubs = None
        self._nsamp = None
        self._npols = None

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype

        # Initialise and generate output blob depending on where it is placed in pipeline
        nstreams = input_shape['nbeams'] if self._after_beamformer else input_shape['nants']

        # Check if number of polarizations is defined in input blob
        npols = 1 if 'npols' not in input_shape.keys() else input_shape['npols']

        self._initialise(npols, input_shape['nsamp'], nstreams, input_shape['nsubs'])

        meta_data = [('npols', self._npols),
                     ('nbeams', nstreams),
                     ('nchans', self._nchans * input_shape['nsubs']),
                     ('nsamp', int(input_shape['nsamp'] / self._nchans))]

        if not self._after_beamformer:
            meta_data[1] = ('nants', input_shape['nants'])

        # Generate output blob
        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id):
                return GPUChannelisedBlob(meta_data, datatype=datatype, device=settings.manager.gpu_device_id)
        else:
            return ChannelisedBlob(meta_data, datatype=datatype)

    def _initialise(self, npols, nsamp, nbeams, nsubs):
        """ Initialise temporary arrays if not already initialised """
        # Update nsamp with value in block
        self._npols = npols
        self._nsamp = nsamp
        self._nbeams = nbeams
        self._nsubs = nsubs

        # Generate filter
        self._generate_filter()

        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id):
                self._filter_gpu = cu.asarray(self._filter)

                # Create temporary array for filtered data
                self._filtered = cu.zeros(
                    (self._npols, self._nbeams, self._nsubs, self._nchans, int(self._nsamp / self._nchans)),
                    dtype=np.complex64)

                # Create temporary input array
                self._temp_input = cu.zeros(
                    (self._npols, self._nbeams, self._nsubs, self._nsamp + self._nchans * self._ntaps),
                    dtype=np.complex64)
        else:
            # Create temporary array for filtered data
            self._filtered = np.zeros(
                (self._npols, self._nbeams, self._nsubs, self._nchans, int(self._nsamp / self._nchans)),
                dtype=np.complex64)

            # Create temporary input array
            self._temp_input = np.zeros(
                (self._npols, self._nbeams, self._nsubs, self._nsamp + self._nchans * self._ntaps),
                dtype=np.complex64)

    def process(self, obs_info, input_data, output_data):
        """
        Perform the channelisation

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # Check if initialised, if not initialise
        nstreams = obs_info['nbeams'] if self._after_beamformer else obs_info['nants']
        npols = 1 if 'npols' not in obs_info.keys() else obs_info['npols']
        if self._filter is None:
            self._initialise(npols, obs_info['nsamp'], nstreams, obs_info['nsubs'])

        # Update parameters
        self._nsamp = obs_info['nsamp']
        self._nsubs = obs_info['nsubs']
        self._nbeams = nstreams

        # Set current output
        self._current_output = output_data

        # Update temporary input array (works for GPU and CPU)
        if self._ntaps != 0:
            self._temp_input[:, :, :, :self._nchans * self._ntaps] = self._temp_input[:, :, :,
                                                                     -self._nchans * self._ntaps:]

        # Channelise
        if settings.manager.use_gpu:
            self.channelise_gpu(input_data, output_data)
        else:
            self.channelise_serial(input_data, output_data)

        # Update observation information
        obs_info['timestamp'] -= timedelta(seconds=(self._ntaps - 1) * self._nchans * obs_info['sampling_time'])
        obs_info['nchans'] = self._nchans * obs_info['nsubs']
        obs_info['nsamp'] //= self._nchans
        obs_info['sampling_time'] *= self._nchans
        obs_info['channel_bandwidth'] /= self._nchans
        obs_info['start_center_frequency'] -= obs_info['channel_bandwidth'] * self._nchans / 2.0

        # Done, return observation information
        return obs_info

    # ------------------------------------------- HELPER FUNCTIONS ---------------------------------------

    def _generate_filter(self):
        """
        Generate FIR filter (Hanning window) for PFB
        :return:
        """

        dx = math.pi / self._nchans
        x = np.array([n * dx - self._ntaps * math.pi / 2 for n in range(self._ntaps * self._nchans)])
        self._filter = np.sinc(self._bin_width_scale * x / math.pi) * np.hanning(self._ntaps * self._nchans)

        # Reverse filter to ease fast computation
        self._filter = self._filter[::-1]

    def _tear_down(self):
        """ Tear down channeliser """
        del self._filter_gpu
        del self._temp_input

    def channelise_gpu(self, input_data, output_data):
        """ Perform channelization on a GPU """
        with cu.cuda.device.Device(settings.manager.gpu_device_id) as d:

            # Format channeliser input depending on where it was placed in pipeline
            # TODO: Handle case where input is not in GPU
            if self._after_beamformer:
                self._temp_input[:, :, :, self._nchans * self._ntaps:] = cu.asarray(input_data)
            else:
                self._temp_input[:, :, :, self._nchans * self._ntaps:] = cu.asarray(
                    np.transpose(input_data, (0, 3, 1, 2)))

            # Call filtering kernel
            nof_spectra = input_data.shape[-2] // self._nchans
            nof_threads = 64
            grid = (math.ceil(nof_spectra / nof_threads), self._nbeams)
            apply_fir_filter_cuda[grid, nof_threads](self._temp_input, self._filter_gpu, self._filtered,
                                                     nof_spectra, self._nchans, self._ntaps)

            # Perform FFTs
            output_data[:] = cupy.squeeze(fftshift(fft(self._filtered, overwrite_x=True, axis=-2), axes=-2))

            # Synchronize
            d.synchronize()

    def channelise_serial(self, input_data, output_data):
        """
        Perform channelisation, serial version
        :return:
        """

        # Format channeliser input depending on where it was placed in pipeline
        if self._after_beamformer:
            self._temp_input[:, :, :, self._nchans * self._ntaps:] = input_data
        else:
            self._temp_input[:, :, :, self._nchans * self._ntaps:] = np.transpose(input_data, (0, 3, 1, 2))

        for p in range(self._npols):
            for b in range(self._nbeams):
                for c in range(self._nsubs):
                    # Apply filter
                    if self._ntaps != 0:
                        apply_fir_filter(self._temp_input[p, b, c, :], self._filter,
                                         self._filtered[p, b, c, :], self._ntaps, self._nchans)
                    else:
                        self._filtered = np.reshape(self._temp_input, (self._npols, self._nbeams, self._nsubs,
                                                                       self._nchans, int(self._nsamp / self._nchans)))

        output_data[:] = np.squeeze(np.fft.fftshift(np.fft.fft(self._filtered, axis=-2), axes=-2))
