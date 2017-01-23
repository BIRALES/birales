import logging

from blobs.correlated_data import CorrelatedBlob
from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.channelised_data import ChannelisedBlob
from pybirales.blobs.dummy_data import DummyBlob
from pybirales.blobs.receiver_data import ReceiverBlob


class Correlator(ProcessingModule):
    """ Correlator module """

    def __init__(self, config, input_blob=None):

        # This module needs an input blob of type dummy, receiver or channeliser
        if type(input_blob) not in [ChannelisedBlob, DummyBlob, ReceiverBlob]:
            raise PipelineError("Correlator: Invalid input data type, should be ChannelisedBlob, "
                                "DummyBlob or ReceiverBlob")

        # Check if we're dealing with channelised data or receiver data
        self._after_channelizer = True if type(input_blob) is ChannelisedBlob else False

        # Sanity checks on configuration
        if {'integration_time'} - set(config.settings()) != set():
            raise PipelineError("Correlator: Missing keys on configuration. (nchans, integration_time, nsamp)")

        # Call superclass initialiser
        super(Correlator, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Correlator"

        # Populate variables
        self._integration_time = config.integration_time
        self._nants = None
        self._nchans = None
        self._nsamp = None

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype

        # Check if integration time is a multiple of nsamp
        if input_shape['nsamp'] % self._config.integration_time != 0:
            raise PipelineError("Correlator: integration time must be a multiple of nsamp")

        # Generate output blob
        return CorrelatedBlob(self._config, [('nsamp', input_shape['nsamp'] / self._config.integration_time),
                                             ('nchans', input_shape['nchans'] if self._after_channelizer else input_shape['nsubs']),
                                             ('nants', input_shape['nants'])],
                               datatype=datatype)

    def process(self, obs_info, input_data, output_data):
        """ Perform channelisation """

        # Update parameters
        self._nsamp = obs_info['nsamp']
        self._nchans = obs_info['nsubs']
        self._nants = obs_info['nants']

        # Re-perform integration time check

        # Update observation information
        logging.info("Correlated data")
