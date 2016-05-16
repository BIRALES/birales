import logging
import os
import pickle
import time

from pybirales.base import settings
from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule
import numpy as np


class Persister(ProcessingModule):
    """ Dummy data generator """

    def __init__(self, config, input_blob=None):

        # Call superclass initialiser
        super(Persister, self).__init__(config, input_blob)

        # Sanity checks on configuration
        if {'directory'} - set(config.settings()) != set():
            raise PipelineError("Persister: Missing keys on configuration. (directory)")

        # Create file
        if config.use_timestamp:
            filepath = os.path.join(config.directory, str(time.now()))
        else:
            if 'filename' not in config.settings():
                raise PipelineError("Persister: filename required when not using timestamp")
            filepath = os.path.join(config.directory, config.filename + '.dat')

        # Open file (if file exists, remove first)
        if os.path.exists(filepath):
            os.remove(filepath)
        self._file = open(filepath, "ab+")

        # Variable to check whether meta file has been written
        self._head_filepath = filepath + '.pkl'
        self._head_written = False

        # Processing module name
        self.name = "Persister"

    def generate_output_blob(self):
        """ Generate output data blob """
        return None

    def process(self, obs_info, input_data, output_data):

        # If head file not written, write it now
        if not self._head_written:
            obs_info['start_center_frequency'] = settings.observation.start_center_frequency
            obs_info['bandwidth'] = settings.observation.bandwidth
            with open(self._head_filepath, 'w') as f:
                pickle.dump(obs_info, f)
            self._head_written = True

        # Transpose data and write to file
        np.save(self._file, input_data.T)
        logging.info("Persisted data")
