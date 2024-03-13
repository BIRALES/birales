import math
from datetime import timedelta

import numba
import numpy as np
from numba import cuda
from pybirales import settings
from pybirales.pipeline.base.cuda_wrapper import try_cuda_jit

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
def apply_fir_filter(data, fir_filter, output, nof_taps, nof_channels):
    """
    Optimised filter function using numpy and numba
    :param data: Input data pointer
    :param fir_filter: Filter coefficients pointer
    :param output: Output data pointer
    :param nof_taps: Number of taps
    :param nof_channels: Number of channels
    """
    nof_spectra = (len(data) - nof_taps * nof_channels) // nof_channels
    for n in range(nof_spectra):
        temp = data[n * nof_channels: n * nof_channels + nof_channels * nof_taps] * fir_filter
        for j in range(1, nof_taps):
            temp[:nof_channels] += temp[j * nof_channels: (j + 1) * nof_channels]
        output[:, n] = temp[:nof_channels]


@try_cuda_jit('void(complex64[:,:,:,:], float64[:], complex64[:,:,:,:,:], int32, int32, int32)', fastmath=True)
def apply_fir_filter_cuda(input_data, fir_filter, output_data, nof_spectra, nof_channels, nof_taps):
    """
    Optimised filter function using numpy and numba
    :param input_data: Input data pointer
    :param fir_filter: Filter coefficients pointer
    :param output_data: Output data pointer
    :param nof_spectra: Number of spectra to process
    :param nof_taps: Number of taps
    :param nof_channels: Number of channels
    """

    # NOTE: This assumes that nof_polarisations and nof_subbands are 1 for the time being
    spectrum = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    stream = cuda.blockIdx.y

    # Get relevant parts of input and output buffers
    input_ptr = input_data[0, stream, 0]
    output_ptr = output_data[0, stream, 0]

    if spectrum >= nof_spectra:
        return

    # Loop across channels and each tap multiplied by the associated filter value to generate
    # the filtered channel response
    for channel in range(nof_channels):
        temp_value = 0
        for tap in range(nof_taps):
            temp_value += input_ptr[spectrum * nof_channels + tap * nof_channels + channel] * fir_filter[tap * nof_channels + channel]
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
        if {'nof_channels', 'nof_taps'} - set(config.settings()) != set():
            raise PipelineError("PPF: Missing keys on configuration. (nof_channels, nof_taps, nof_samples, nof_beams)")
        self._bin_width_scale = 1.0
        self._nof_channels = config.nof_channels
        self._nof_taps = config.nof_taps

        # Call superclass initializer
        super(PFB, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Channeliser"

        # Variable below will be populated in generate_output_blob
        self._filter = None
        self._filter_gpu = None
        self._filtered = None
        self._temp_input = None
        self._nof_beams = None
        self._nof_subbands = None
        self._nof_samples = None
        self._nof_polarisations = None

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype

        # Initialise and generate output blob depending on where it is placed in pipeline
        nof_streams = input_shape['nof_beams'] if self._after_beamformer else input_shape['nof_antennas']

        # Check if number of polarizations is defined in input blob
        nof_polarisations = 1 if 'nof_polarisations' not in input_shape.keys() else input_shape['nof_polarisations']

        self._initialise(nof_polarisations, input_shape['nof_samples'], nof_streams, input_shape['nof_subbands'])

        meta_data = [('nof_polarisations', self._nof_polarisations),
                     ('nof_beams', nof_streams),
                     ('nof_channels', self._nof_channels * input_shape['nof_subbands']),
                     ('nof_samples', int(input_shape['nof_samples'] / self._nof_channels))]

        if not self._after_beamformer:
            meta_data[1] = ('nof_antennas', input_shape['nof_antennas'])

        # Generate output blob
        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id):
                return GPUChannelisedBlob(meta_data, datatype=datatype, device=settings.manager.gpu_device_id)
        else:
            return ChannelisedBlob(meta_data, datatype=datatype)

    def _initialise(self, nof_polarisations, nof_samples, nof_beams, nof_subbands):
        """ Initialise temporary arrays if not already initialised """
        # Update nof_samples with value in block
        self._nof_polarisations = nof_polarisations
        self._nof_samples = nof_samples
        self._nof_beams = nof_beams
        self._nof_subbands = nof_subbands

        # Generate filter
        self._generate_filter()

        if settings.manager.use_gpu:
            with cu.cuda.Device(settings.manager.gpu_device_id):
                self._filter_gpu = cu.asarray(self._filter)

                # Create temporary array for filtered data
                self._filtered = cu.zeros(
                    (self._nof_polarisations, self._nof_beams, self._nof_subbands, self._nof_channels, int(self._nof_samples / self._nof_channels)),
                    dtype=np.complex64)

                # Create temporary input array
                self._temp_input = cu.zeros(
                    (self._nof_polarisations, self._nof_beams, self._nof_subbands, self._nof_samples + self._nof_channels * self._nof_taps),
                    dtype=np.complex64)
        else:
            # Create temporary array for filtered data
            self._filtered = np.zeros(
                (self._nof_polarisations, self._nof_beams, self._nof_subbands, self._nof_channels, int(self._nof_samples / self._nof_channels)),
                dtype=np.complex64)

            # Create temporary input array
            self._temp_input = np.zeros(
                (self._nof_polarisations, self._nof_beams, self._nof_subbands, self._nof_samples + self._nof_channels * self._nof_taps),
                dtype=np.complex64)

    def process(self, obs_info, input_data, output_data):
        """
        Perform the channelisation

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        if obs_info['stop_pipeline_at'] != -1:
            return

        # Check if initialised, if not initialise
        nof_streams = obs_info['nof_beams'] if self._after_beamformer else obs_info['nof_antennas']
        nof_polarisations = 1 if 'nof_polarisations' not in obs_info.keys() else obs_info['nof_polarisations']
        if self._filter is None:
            self._initialise(nof_polarisations, obs_info['nof_samples'], nof_streams, obs_info['nof_subbands'])

        # Update parameters
        self._nof_samples = obs_info['nof_samples']
        self._nof_subbands = obs_info['nof_subbands']
        self._nof_beams = nof_streams

        # Update temporary input array (works for GPU and CPU)
        if self._nof_taps != 0:
            self._temp_input[:, :, :, :self._nof_channels * self._nof_taps] = self._temp_input[:, :, :,
                                                                     -self._nof_channels * self._nof_taps:]

        # Channelise
        if settings.manager.use_gpu:
            self.channelise_gpu(input_data, output_data)
        else:
            self.channelise_serial(input_data, output_data)

        # Update observation information
        obs_info['timestamp'] -= timedelta(seconds=(self._nof_taps - 1) * self._nof_channels * obs_info['sampling_time'])
        obs_info['nof_channels'] = self._nof_channels * obs_info['nof_subbands']
        obs_info['nof_samples'] //= self._nof_channels
        obs_info['sampling_time'] *= self._nof_channels
        obs_info['channel_bandwidth'] /= self._nof_channels
        obs_info['start_center_frequency'] -= obs_info['channel_bandwidth'] * self._nof_channels / 2.0

        # Done, return observation information
        return obs_info

    # ------------------------------------------- HELPER FUNCTIONS ---------------------------------------

    def _generate_filter(self):
        """
        Generate FIR filter (Hanning window) for PFB
        :return:
        """

        dx = math.pi / self._nof_channels
        x = np.array([n * dx - self._nof_taps * math.pi / 2 for n in range(self._nof_taps * self._nof_channels)])
        self._filter = np.sinc(self._bin_width_scale * x / math.pi) * np.hanning(self._nof_taps * self._nof_channels)

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
                self._temp_input[:, :, :, self._nof_channels * self._nof_taps:] = cu.asarray(input_data)
            else:
                self._temp_input[:, :, :, self._nof_channels * self._nof_taps:] = cu.asarray(
                    np.transpose(input_data, (0, 3, 1, 2)))

            # Call filtering kernel
            nof_spectra = int(self._nof_samples // self._nof_channels)
            nof_threads = 64
            grid = (math.ceil(nof_spectra / nof_threads), self._nof_beams)
            apply_fir_filter_cuda[grid, nof_threads](self._temp_input, self._filter_gpu, self._filtered,
                                                     nof_spectra, self._nof_channels, self._nof_taps)

            # Perform FFTs
            output_data[:] = cu.squeeze(fftshift(fft(self._filtered, overwrite_x=True, axis=-2), axes=-2))

            # Synchronize
            d.synchronize()

    def channelise_serial(self, input_data, output_data):
        """
        Perform channelisation, serial version
        :return:
        """

        # Format channeliser input depending on where it was placed in pipeline
        if self._after_beamformer:
            self._temp_input[:, :, :, self._nof_channels * self._nof_taps:] = input_data
        else:
            self._temp_input[:, :, :, self._nof_channels * self._nof_taps:] = np.transpose(input_data, (0, 3, 1, 2))

        for p in range(self._nof_polarisations):
            for b in range(self._nof_beams):
                for c in range(self._nof_subbands):
                    # Apply filter
                    if self._nof_taps != 0:
                        apply_fir_filter(self._temp_input[p, b, c, :], self._filter,
                                         self._filtered[p, b, c, :], self._nof_taps, self._nof_channels)
                    else:
                        self._filtered = np.reshape(self._temp_input,
                                                    (self._nof_polarisations, self._nof_beams, self._nof_subbands,
                                                     self._nof_channels, int(self._nof_samples / self._nof_channels)))

        output_data[:] = np.squeeze(np.fft.fftshift(np.fft.fft(self._filtered, axis=-2), axes=-2))