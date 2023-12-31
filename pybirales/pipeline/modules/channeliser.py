import math
from datetime import timedelta
from multiprocessing.pool import ThreadPool

import numba
import numpy as np

from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.beamformed_data import BeamformedBlob
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.blobs.dummy_data import DummyBlob
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob


@numba.jit(nopython=True, nogil=True)
def apply_fir_filter(data, fir_filter, output, ntaps, nchans):
    """
    Optimised filter function using numpy and numba
    :param data: Input data pointer
    :param fir_filter: Filter coefficients pointerself
    :param output: Output data pointer
    :param ntaps: Number of taps
    :param nchans: Number of channels
    """
    nof_spectra = (len(data) - ntaps * nchans) / nchans
    for n in range(nof_spectra):
        temp = data[n * nchans: n * nchans + nchans * ntaps] * fir_filter
        for j in range(1, ntaps):
            temp[:nchans] += temp[j * nchans: (j + 1) * nchans]
        output[:, n] = temp[:nchans]


class PFB(ProcessingModule):
    """ PPF processing module """

    def __init__(self, config, input_blob=None):

        # This module needs an input blob of type dummy
        if type(input_blob) not in [BeamformedBlob, DummyBlob, ReceiverBlob]:
            raise PipelineError("PFB: Invalid input data type, should be BeamformedBlob, DummyBlob or ReceiverBlob")

        # Check if we're dealing with beamformed data or receiver data
        self._after_beamformer = True if type(input_blob) is BeamformedBlob else False

        # Sanity checks on configuration
        if {'nchans', 'ntaps'} - set(config.settings()) != set():
            raise PipelineError("PPF: Missing keys on configuration. (nchans, ntaps, nsamp, nbeams)")
        self._bin_width_scale = 1.0
        self._nchans = config.nchans
        self._ntaps = config.ntaps
        self._use_numba = config.use_numba

        self._nthreads = 2
        if 'nthreads' in config.settings():
            self._nthreads = config.nthreads

        # Call superclass initialiser
        super(PFB, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Channeliser"

        # Create thread pool for parallel PFB
        self._thread_pool = None
        if self._use_numba:
            self._thread_pool = ThreadPool(self._nthreads)

        # Variable below will be populated in generate_output_blob
        self._filter = None
        self._filtered = None
        self._temp_input = None
        self._current_output = None
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
        return ChannelisedBlob(self._config, meta_data, datatype=datatype)

    def _initialise(self, npols, nsamp, nbeams, nsubs):
        """ Initialise temporary arrays if not already initialised """
        # Update nsamp with value in block
        self._npols = npols
        self._nsamp = nsamp
        self._nbeams = nbeams
        self._nsubs = nsubs

        # Generate filter
        self._generate_filter()

        # Create temporary array for filtered data
        self._filtered = np.zeros(
            (self._npols, self._nbeams, self._nsubs, self._nchans, int(self._nsamp / self._nchans)),
            dtype=np.complex64)

        # Create temporary input array
        self._temp_input = np.zeros((self._npols, self._nbeams, self._nsubs, self._nsamp + self._nchans * self._ntaps),
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

        # Update temporary input array
        self._temp_input[:, :, :, :self._nchans * self._ntaps] = self._temp_input[:, :, :, -self._nchans * self._ntaps:]

        # Format channeliser input depending on where it was placed in pipeline
        if self._after_beamformer:
            self._temp_input[:, :, :, self._nchans * self._ntaps:] = input_data
        else:
            # TODO: Fix this transpose
            self._temp_input[:, :, :, self._nchans * self._ntaps:] = np.transpose(input_data, (0, 3, 1, 2))

        # Channelise
        if self._use_numba:
            self.channelise_parallel()
        else:
            self.channelise_serial()

        # Update observation information
        obs_info['timestamp'] -= timedelta(seconds=(self._ntaps - 1) * self._nchans * obs_info['sampling_time'])
        print(timedelta(seconds=(self._ntaps - 1) * self._nchans * obs_info['sampling_time']))
        print((self._ntaps - 1), self._nchans, obs_info['sampling_time'])
        obs_info['nchans'] = self._nchans * obs_info['nsubs']
        obs_info['nsamp'] //= self._nchans
        obs_info['sampling_time'] *= self._nchans

        obs_info['channel_bandwidth'] /= self._nchans
        obs_info['start_center_frequency'] -= obs_info['channel_bandwidth'] * self._nchans / 2.0

        # Debug info
        # print(f'{self._iter_count} Time: {obs_info["timestamp"]:%Y-%m-%d %H:%M:%S.%f}')
        # print(obs_info['start_center_frequency'])
        # print('Channeliser {:0.7f}s shape:{}, iter:{}'.format(obs_info['sampling_time'], np.shape(output_data), self._iter_count))
        # print('Channeliser bandwidth from: {} Mhz to {} Mhz'.format(obs_info['start_center_frequency'], obs_info['start_center_frequency'] + obs_info['channel_bandwidth']*8192))

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
        if self._use_numba:
            self._thread_pool.close()

    def channelise_thread(self, beam):
        """
        Perform channelisation, to be used with ThreadPool

        :param beam: Beam number associated with call
        :return:
        """
        for p in range(self._npols):
            for c in range(self._nsubs):
                # Apply filter
                apply_fir_filter(self._temp_input[p, beam, c, :], self._filter,
                                 self._filtered[p, beam, c, :], self._ntaps, self._nchans)

                # Fourier transform and save output
                self._current_output[p, beam, c * self._nchans: (c + 1) * self._nchans] = \
                    np.fft.fftshift(np.fft.fft(self._filtered[p, beam, c, :], axis=0), axes=0)

    def channelise_parallel(self):
        """
       Perform channelisation, parallel version
       :return:
       """
        self._thread_pool.map(self.channelise_thread, range(self._nbeams))

    def channelise_serial(self):
        """
        Perform channelisation, serial version
        :return:
        """
        for p in range(self._npols):
            for b in range(self._nbeams):
                for c in range(self._nsubs):
                    # Apply filter
                    apply_fir_filter(self._temp_input[p, b, c, :], self._filter,
                                     self._filtered[p, b, c, :], self._ntaps, self._nchans)

                    # Fourier transform and save output
                    # Changed to work with TPM
                    self._current_output[p, b, c * self._nchans: (c + 1) * self._nchans] = \
                        np.fft.fftshift(np.fft.fft(self._filtered[p, b, c, :], axis=0), axes=0)

                    # self._current_output[p, b, c * self._nchans: (c + 1) * self._nchans] = np.flipud(
                    #     np.fft.fft(self._filtered[p, b, c, :], axis=0))

                    # self._current_output[p, b, c * self._nchans: (c + 1) * self._nchans] = np.flipud(
                    # np.fft.fftshift(np.fft.fft(self._filtered[p, b, c, :], axis=0), axes=0))
