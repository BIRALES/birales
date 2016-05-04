import numpy as np
import time

from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.base.util import atomic_print
from pybirales.blobs.beamformed_data import BeamformedBlob
from pybirales.blobs.dummy_data import DummyBlob


class Beamformer(ProcessingModule):
    """ Beamformer processing module """

    def __init__(self, config, input_blob=None):
        # This module needs an input blob of type channelised
        if type(input_blob) is not DummyBlob:
            raise PipelineError("Baeamformer: Invalid input data type, should be DummyBlob")

        # Sanity checks on configuration
        if 'nbeams' not in config.keys():
            raise PipelineError("Beamformer: Missing keys on configuration. Number of beams required")

        self._nbeams = config['nbeams']

        # Call superclass initialiser
        super(Beamformer, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Channeliser"

    def generate_output_blob(self):
        """ Generate output data blob """
        input_shape = dict(self._input.shape)
        datatype = self._input.datatype
        return BeamformedBlob(self._config, [('nbeams', self._nbeams),
                                             ('nchans', input_shape['nchans']),
                                             ('nsamp', input_shape['nsamp'])],
                              datatype=datatype)

    def process(self, obs_info, input_data, output_data):
        time.sleep(0.01)
        output_data[:] = np.reshape(np.sum(input_data, axis=2), (1, 1, 524288)).repeat(32, 0)
        atomic_print("Beamformed data")
