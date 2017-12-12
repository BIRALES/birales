import os
import pickle
import time
import struct
from pybirales import settings
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob
from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
import numpy as np
import fadvise


class RawPersister(ProcessingModule):
    """ Raw data persister """

    def __init__(self, config, input_blob=None):
        # Call superclass initialiser
        super(RawPersister, self).__init__(config, input_blob)

        # Sanity checks on configuration
        if {'directory'} - set(config.settings()) != set():
            raise PipelineError("Persister: Missing keys on configuration. (directory)")

        # Create directory if it doesn't exist
        directory = settings.persisters.directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create file
        if config.use_timestamp:
            file_path = os.path.join(directory, "%s_%s" % (self._config.filename_suffix, str(time.time())))
        else:
            if 'filename' not in config.settings():
                raise PipelineError("Raw Persister: Filename required when not using timestamp")
            file_path = os.path.join(directory, self._config.filename_suffix + '.dat')

        # Open file (if file exists, remove first)
        if os.path.exists(file_path):
            os.remove(file_path)

        self._file = open(file_path, "wb+")
        fadvise.set_advice(self._file, fadvise.POSIX_FADV_SEQUENTIAL)

        # Variable to check whether meta file has been written
        self._head_filepath = file_path + '.pkl'
        self._head_written = False

        # Processing module name
        self.name = "RawPersister"

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return ReceiverBlob(self._config, self._input.shape, datatype=np.complex64)

    def process(self, obs_info, input_data, output_data):

        # If head file not written, write it now
        if not self._head_written:
            with open(self._head_filepath, 'wb') as f:
                pickle.dump(obs_info.get_dict(), f)
            self._head_written = True

        # Copy input data to output data
        if output_data is not None:
            output_data[:] = input_data

        # Expand complex data, transpose and save to file
        data = input_data[0, 0, :, :]
        data = np.array([data.real, data.imag])
        data = np.transpose(data, (2,1,0))
        data = np.transpose(data, (1, 0, 2))
        data = data.ravel()

        self._file.write(struct.pack('f' * len(data), *data))
        self._file.flush()
