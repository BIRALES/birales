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
        if {'nants', 'nsamp', 'nchans', 'nbits', 'complex'} - set(config.keys()) != set():
            raise PipelineError("DummyDataGenerator: Missing keys on configuration "
                                "(nants, nsamp, nchans, nbits, complex")
        self._nants = config['nants']
        self._nsamp = config['nsamp']
        self._nchans = config['nchans']
        self._nbits = config['nbits']
        self._complex = config['complex']

        # Define data type
        if self._nbits == 64 and self._complex:
            self._datatype = np.complex64
        else:
            raise PipelineError("DummyDataGenerator: Unsupported datatype (bits, complex)")

        # Call superclass initialiser
        super(DummyDataGenerator, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Generator  "
        self._count = 0

    def generate_output_blob(self):
        """ Generate output data blob """

        # Generate blob
        return DummyBlob(self._config, [('nchans', self._nchans),
                                        ('nsamp', self._nsamp),
                                        ('nants', self._nants)],
                         datatype=self._datatype)

    def process(self, obs_info, input_data, output_data):
        time.sleep(0.1)
        self._count += 1
        output_data[:] = np.ones((self._nchans, self._nsamp, self._nchans), dtype=self._datatype) * self._count
        atomic_print("Generated data")
