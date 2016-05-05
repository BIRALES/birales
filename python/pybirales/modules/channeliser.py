import numpy as np
import logging

import math

from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.beamformed_data import BeamformedBlob
from pybirales.blobs.channelised_data import ChannelisedBlob

class PFB(ProcessingModule):
    """ PPF processing module """

    def __init__(self, config, input_blob=None):

        # This module needs an input blob of type dummy
        if type(input_blob) is not BeamformedBlob:
            raise PipelineError("PPF: Invalid input data type, should be BeamformedBlob")

        # Sanity checks on configuration
        if {'nchans', 'ntaps'} - set(config.settings()) != set():
            raise PipelineError("PPF: Missing keys on configuration. (nchans, ntaps)")
        self._nchans = config.nchans
        self._ntaps = config.ntaps

        # Call superclass initialiser
        super(PFB, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Channeliser"

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype
        return ChannelisedBlob(self._config, [('nbeams', input_shape['nbeams']),
                                              ('nsamp', input_shape['nsamp'] / self._nchans),
                                              ('nchans', self._nchans)],
                               datatype=datatype)

    def process(self, obs_info, input_data, output_data):
        # Perform channelisation
        if self._no_output:
            output = np.reshape(input_data, (obs_info['nbeams'], obs_info['nsamp']/ self._nchans, self._nchans))
        else:
            output_data[:] = np.reshape(input_data, (obs_info['nbeams'], obs_info['nsamp'] / self._nchans, self._nchans))

        # Update observation information
        obs_info['nchans'] = self._nchans * obs_info['nchans']
        obs_info['channeliser'] = "Done"
        print obs_info
        logging.info("Channelised data")

    def _generate_filter(self):
        """ Generate FIR filter (Hanning window) for PFB """
        dx = math.pi / self._nchans
        X = np.array([n * dx - self._ntaps * math.pi / 2 for n in xrange(self._ntaps * self._nchans)])
        self._filter = np.sinc(self._bin_width_scale * X / math.pi) * np.hanning(self._ntaps * self._nchans)
