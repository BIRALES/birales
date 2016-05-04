import numpy as np
import time

from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
from pybirales.base.util import atomic_print
from pybirales.blobs.dummy_data import DummyBlob


class DummyDataGenerator(ProcessingModule):
    """ Dummy data generator """

    def __init__(self, config, input_blob=None):

        # This module does not need an input_blob
        if input_blob is not None:
            raise PipelineError("DummyDataGenerator: Invalid input data type, should be None")

        # Sanity checks on configuration
        if {'nants', 'nsamp', 'nchans'} - set(config.keys()) != set():
            raise PipelineError("DummyDataGenerator: Missing keys on configuration. Number of samples, "
                                "antennas and channels required")
        self._nants = config['nants']
        self._nsamp = config['nsamp']
        self._nchans = config['nchans']

        # Call superclass initialiser
        super(DummyDataGenerator, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Generator  "
        self._count = 0

    def generate_output_blob(self):
        """ Generate output data blob """
        return DummyBlob(self._config, [('nchans', self._nchans),
                                        ('nants', self._nants),
                                        ('nsamp', self._nsamp)])

    def process(self, obs_info, input_data, output_data):
        time.sleep(0.01)
        self._count += 1
        if self._count % 5 == 0:
            time.sleep(0.1)
        output_data = np.zeros((self._nants, self._nsamp, self._nchans))
        atomic_print("Generated data")
