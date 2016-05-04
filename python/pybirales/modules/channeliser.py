import numpy as np
import time

from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.base.util import atomic_print
from pybirales.blobs.beamformed_data import BeamformedBlob
from pybirales.blobs.channelised_data import ChannelisedBlob


class PPF(ProcessingModule):
    """ PPF processing module """

    def __init__(self, config, input_blob=None):

        # This module needs an input blob of type dummy
        if type(input_blob) is not BeamformedBlob:
            raise PipelineError("PPF: Invalid input data type, should be BeamformedBlob")

        # Sanity checks on configuration
        if 'nchans' not in config.keys():
            raise PipelineError("PPF: Missing keys on configuration. Number of channels required")
        self._nchans = config['nchans']

        # Call superclass initialiser
        super(PPF, self).__init__(config, input_blob)

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
        time.sleep(0.01)
        output_data[:] = np.reshape(input_data, (32, 524288 / self._nchans, self._nchans))
        atomic_print("Channelised data")
