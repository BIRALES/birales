import numpy as np
import time

from pybirales.base.definitions import PipelineError, ObservationInfo
from pybirales.base.processing_module import ProcessingModule
from pybirales.blobs.dummy_data import DummyBlob


class Terminator(ProcessingModule):
    """ Dummy data generator """

    def __init__(self, config, input_blob=None):

        # Call superclass initialiser
        super(Terminator, self).__init__(config, input_blob)

        # Processing module name
        self.name = "Terminator"

    def generate_output_blob(self):
        """ Generate output data blob """
        return None

    def process(self, obs_info, input_data, output_data):
        return obs_info
