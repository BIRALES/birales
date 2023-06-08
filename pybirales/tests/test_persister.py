import cupy
import numpy as np

from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.base.processing_module import ProcessingModule


class TestPersister(ProcessingModule):

    def __init__(self, config, input_blob=None):
        # Call superclass initialiser
        super(TestPersister, self).__init__(config, input_blob)

        # Counter
        self._counter = 0

        # Processing module name
        self.name = "TestPersister"

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return ChannelisedBlob(self._input.shape, datatype=np.complex64)

    def process(self, obs_info, input_data, output_data):
        """

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """
        if type(input_data) == cupy.ndarray:
            input_data = cupy.asnumpy(input_data)

        np.save("test.npy", input_data)
        output_data[:] = input_data
