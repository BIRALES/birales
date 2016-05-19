import numpy as np
import logging
from multiprocessing.pool import ThreadPool
import numba
import math
from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.beamformed_data import BeamformedBlob
from pybirales.blobs.channelised_data import ChannelisedBlob
from pybirales.blobs.dummy_data import DummyBlob
from pybirales.blobs.receiver_data import ReceiverBlob


@numba.jit(nopython=True, nogil=True)
def apply_fir_filter(data, fir_filter, output, ntaps, nchans):
    """ Optimised filter function using numpy and numba
    :param data: Input data pointer
    :param fir_filter: Filter coefficients pointer
    :param output: Output data pointer
    :param ntaps: Number of taps
    :param nchans: Number of channels """

    nof_spectra = (len(data) - ntaps * nchans) / nchans
    for n in xrange(nof_spectra):
        temp = data[n * nchans: n * nchans + nchans * ntaps] * fir_filter
        for j in xrange(1, ntaps):
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

        self._nthreads = 2
        if 'nthreads' in config.settings():
            self._nthreads = config.nthreads

        # Call superclass initialiser
        super(PFB, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Channeliser"

        # Create thread pool for parallel PFB
        self._thread_pool = ThreadPool(self._nthreads)

        # Variable below will be populated in generate_output_blob
        self._filter = None
        self._filtered = None
        self._temp_input = None
        self._current_output = None
        self._nbeams = None
        self._nsubs = None
        self._nsamp = None

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype

        # Initialise and generate output blob depending on where it is placed in pipeline
        nstreams = input_shape['nbeams'] if self._after_beamformer else input_shape['nants']

        self._initialise(input_shape['nsamp'], nstreams, input_shape['nsubs'])

        # Generate output blob
        return ChannelisedBlob(self._config, [('nbeams', nstreams),
                                              ('nchans', self._nchans * input_shape['nsubs']),
                                              ('nsamp', input_shape['nsamp'] / self._nchans)],
                               datatype=datatype)

    def _initialise(self, nsamp, nbeams, nsubs):
        """ Initialise temporary arrays if not already initialised """
        # Update nsamp with value in block
        self._nsamp = nsamp
        self._nbeams = nbeams
        self._nsubs = nsubs

        # Generate filter
        self._generate_filter()

        # Create temporary array for filtered data
        self._filtered = np.zeros((self._nbeams, self._nsubs, self._nchans, self._nsamp / self._nchans),
                                  dtype=np.complex64)

        # Create temporary input array
        self._temp_input = np.zeros((self._nbeams, self._nsubs, self._nsamp + self._nchans * self._ntaps),
                                    dtype=np.complex64)

    def process(self, obs_info, input_data, output_data):
        """ Perform channelisation """

        # Check if initialised, if not initialise
        nstreams = obs_info['nbeams'] if self._after_beamformer else obs_info['nants']
        if self._filter is None:
            self._initialise(obs_info['nsamp'], nstreams, obs_info['nsubs'])

        # Update parameters
        self._nsamp = obs_info['nsamp']
        self._nsubs = obs_info['nsubs']
        self._nbeams = nstreams

        # Set current output
        self._current_output = output_data

        # Update temporary input array
        self._temp_input[:, :, :self._nchans * self._ntaps] = self._temp_input[:, :, -self._nchans * self._ntaps:]

        # Format channeliser input depending on where it was placed in pipeline
        if self._after_beamformer:
            self._temp_input[:, :, self._nchans * self._ntaps:] = input_data
        else:
            self._temp_input[:, :, self._nchans * self._ntaps:] = np.transpose(input_data, (2, 0, 1))

        # Channelise
        self.channelise_parallel()

        # Update observation information
        obs_info['nchans'] = self._nchans * obs_info['nsubs']
        logging.info("Channelised data")

    # ------------------------------------------- HELPER FUNCTIONS ---------------------------------------

    def _generate_filter(self):
        """ Generate FIR filter (Hanning window) for PFB """
        dx = math.pi / self._nchans
        X = np.array([n * dx - self._ntaps * math.pi / 2 for n in range(self._ntaps * self._nchans)])
        self._filter = np.sinc(self._bin_width_scale * X / math.pi) * np.hanning(self._ntaps * self._nchans)

        # Reverse filter to ease fast computation
        self._filter = self._filter[::-1]

    def channelise_thread(self, beam):
        """ Perform channelisation, to be used with ThreadPool
        :param beam: Beam number associated with call  """
        for c in range(self._nsubs):
            # Apply filter
            apply_fir_filter(self._temp_input[beam, c, :], self._filter,
                             self._filtered[beam, c, :], self._ntaps, self._nchans)

            # Fourier transform and save output
            self._current_output[beam, c * self._nchans: (c + 1) * self._nchans] = \
                np.abs(np.fft.fft(self._filtered[beam, c, :], axis=0))

    def channelise_parallel(self):
        """ Perform channelisation, parallel version """
        self._thread_pool.map(self.channelise_thread, range(self._nbeams))

    def channelise(self):
        """ Perform channelisation, serial version """
        for b in range(self._nbeams):
            for c in range(self._nsubs):
                # Apply filter
                apply_fir_filter(self._temp_input[b, c, :], self._filter,
                                 self._filtered[b, c, :], self._ntaps, self._nchans)
               # self._filtered[b, c, :] = self._temp_input[b, c, :]

                # Fourier transform and save output
                self._current_output[b, c * self._nchans: (c + 1) * self._nchans] = \
                    np.abs(np.fft.fft(self._filtered[b, c, :], axis=0))
