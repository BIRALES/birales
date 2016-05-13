import logging
import numpy as np
import time

from pybirales.base.definitions import PipelineError, ObservationInfo
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.dummy_data import DummyBlob


class DummyDataGenerator(ProcessingModule):
    """ Dummy data generator """

    def __init__(self, config, input_blob=None):

        # This module does not need an input_blob
        if input_blob is not None:
            raise PipelineError("DummyDataGenerator: Invalid input data type, should be None")

        # Sanity checks on configuration
        if {'nants', 'nsamp', 'nsubs', 'nbits', 'complex'} - set(config.settings()) != set():
            raise PipelineError("DummyDataGenerator: Missing keys on configuration "
                                "(nants, nsamp, nsub, nbits, complex")
        self._nants = config.nants
        self._nsamp = config.nsamp
        self._nsubs = config.nsubs
        self._nbits = config.nbits
        self._complex = config.complex

        # Define data type
        if self._nbits == 64 and self._complex:
            self._datatype = np.complex64
        else:
            raise PipelineError("DummyDataGenerator: Unsupported datatype (bits, complex)")

        # Call superclass initialiser
        super(DummyDataGenerator, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Generator"

    def generate_output_blob(self):
        """ Generate output data blob """

        # Generate blob
        return DummyBlob(self._config, [('nsubs', self._nsubs),
                                        ('nsamp', self._nsamp),
                                        ('nants', self._nants)],
                         datatype=self._datatype)

    def process(self, obs_info, input_data, output_data):

        # Perform operations
        output_data[:] = np.ones((self._nsubs, self._nsamp, self._nsubs), dtype=self._datatype) * 1.5

        # Create observation information
        obs_info = ObservationInfo()
        obs_info['sampling_time'] = 0.0
        obs_info['timestamp'] = 0.0
        obs_info['nsubs'] = self._nsubs
        obs_info['nsamp'] = self._nsamp

        logging.info("Generated data")

        return obs_info
