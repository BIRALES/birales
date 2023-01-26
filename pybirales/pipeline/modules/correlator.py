import logging

import numba
import numpy as np

from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.blobs.correlated_data import CorrelatedBlob
from pybirales.pipeline.blobs.dummy_data import DummyBlob
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob


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

                    # Original correlation function - not numba compatible
                    # output_data[i, sc, baseline, 0] = np.correlate(
                    #     input_data[0, c, antenna1, i * integrations:(i + 1) * integrations],
                    #     input_data[0, c, antenna2, i * integrations:(i + 1) * integrations])

                baseline += 1

        sc += 1


class Correlator(ProcessingModule):
    """ Correlator module """

    def __init__(self, config, input_blob=None):

        self._validate_data_blob(input_blob, valid_blobs=[ReceiverBlob, DummyBlob, ChannelisedBlob])

        # Check if we're dealing with channelised data or receiver data
        self._after_channelizer = True if type(input_blob) is ChannelisedBlob else False

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
        self._current_input = None
        self._current_output = None
        self._nbaselines = None
        self._nstokes = None
        self._nsamp = None
        self._nchans = None
        self._nants = None
        self._npols = None

        self._channel_start = None
        self._channel_end = None

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
        blob_type = CorrelatedBlob
        nchans = input_shape['nchans'] if self._after_channelizer else input_shape['nsubs']
        if self._after_channelizer:
            blob_type = ChannelisedBlob

            if hasattr(settings.correlator, 'channel_start') and hasattr(settings.correlator, 'channel_end'):
                nchans = settings.correlator.channel_end - settings.correlator.channel_start

        return blob_type(self._config, [('nsamp', input_shape['nsamp'] / self._config.integrations),
                                        ('nchans', nchans),
                                        ('baselines', baselines),
                                        ('stokes', nstokes)],
                         datatype=datatype)

    def process(self, obs_info, input_data, output_data):
        """ Perform correlation """

        # Update parameters
        self._nsamp = obs_info['nsamp']
        self._nchans = obs_info['nsubs']
        self._nants = obs_info['nants']
        self._npols = obs_info['npols']

        self._nbaselines = int(0.5 * ((obs_info['nants'] ** 2) - obs_info['nants']))
        self._nstokes = obs_info['npols'] ** 2

        # Integration check
        if self._nsamp % self._integrations != 0:
            logging.warning("Number of integration not factor of number of samples, skipping buffer")
            return

        if self._after_channelizer:
            # In case when the number of channels if given, use this.
            # Used when the preceding module is a Channeliser
            self._nchans = obs_info['nchans']
            self._current_input = np.transpose(input_data, (0, 2, 1, 3)).copy()
        else:
            # Transpose the data so that we can parallelize over frequency
            self._current_input = np.transpose(input_data, (0, 1, 3, 2)).copy()

        # Perform correlation
        if self._npols == 1:
            if self._iter_count == 1:
                if hasattr(settings.correlator, 'channel_start') and hasattr(settings.correlator, 'channel_end'):
                    self._channel_start = settings.correlator.channel_start
                    self._channel_end = settings.correlator.channel_end
                else:
                    self._channel_start = 0
                    self._channel_end = self._nchans

            correlate(self._current_input, output_data, self._channel_start, self._channel_end,
                      self._nants, self._integrations, self._nsamp)

        obs_info['nsamp'] = self._nsamp // self._integrations
        obs_info['sampling_time'] *= self._integrations
        obs_info['nbaselines'] = self._nbaselines
        obs_info['nstokes'] = self._nstokes

        return obs_info
