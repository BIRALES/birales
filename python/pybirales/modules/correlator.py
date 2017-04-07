import numpy as np
import logging

from pybirales.blobs.correlated_data import CorrelatedBlob
from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.channelised_data import ChannelisedBlob
from pybirales.blobs.receiver_data import ReceiverBlob
from pybirales.blobs.dummy_data import DummyBlob


class Correlator(ProcessingModule):
    """ Correlator module """

    def __init__(self, config, input_blob=None):

        # This module needs an input blob of type dummy, receiver or channeliser
        if type(input_blob) not in [ReceiverBlob, DummyBlob, ChannelisedBlob]:
            raise PipelineError("Correlator: Invalid input data type, should be ChannelisedBlob, "
                                "DummyBlob or ReceiverBlob")

        # Check if we're dealing with channelised data or receiver data
        self._after_channelizer = True if type(input_blob) is ChannelisedBlob else False

        # Sanity checks on configuration
        if {'integration'} - set(config.settings()) != set():
            raise PipelineError("Correlator: Missing keys on configuration. (integration)")

        # Call superclass initialiser
        super(Correlator, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Correlator"

        # Populate variables
        self._integration = config.integration

        # Define parameters
        self._current_input = None
        self._current_output = None
        self._nsamp = None
        self._nchans = None
        self._nants = None
        self._npols = None

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype

        # Check if integration time is a multiple of nsamp
        if input_shape['nsamp'] % self._config.integration != 0:
            raise PipelineError("Correlator: integration time must be a multiple of nsamp")

        # Generate output blob
        return CorrelatedBlob(self._config, [('nchans',
                                              input_shape['nchans'] if self._after_channelizer else input_shape[
                                                  'nsubs']),
                                             ('nsamp', input_shape['nsamp'] / self._config.integration),
                                             ('baselines', (input_shape['nbeams'] ** 2) / 2),
                                             ('stokes', 4)],
                              datatype=datatype)

    def process(self, obs_info, input_data, output_data):
        """ Perform channelisation """

        # Update parameters
        self._nsamp = obs_info['nsamp']
        self._nchans = obs_info['nchans']
        self._nants = obs_info['nants']
        self._npols = obs_info['npols']

        # TODO: Re-perform integration time check

        # Transpose the data so the we can parallelise over frequency
        self._current_input = np.transpose(input_data, (2, 1, 0, 3))
        self._current_input = np.transpose(self._current_input, (0, 2, 1, 3))

        # Point to output
        self._current_output = output_data

        # Perform correlation
        for c in range(self._nchans):
            for p in range(self._npols):
                for a1 in range(self._nants):
                    for a2 in range(a1, self._nants):
                        np.correlate(self._current_input[c, p, a1, :], self._current_input[c, p, a2, :])

        # Update observation information
        logging.info("Correlated data")
