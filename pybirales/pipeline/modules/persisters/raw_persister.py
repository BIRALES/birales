import datetime
import logging
import os
import pickle
import struct
import time

# import fadvise
import numpy as np

from pybirales import settings
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob


class RawPersister(ProcessingModule):
    """ Raw data persister """

    def __init__(self, config, input_blob=None):
        # Call superclass initialiser
        super(RawPersister, self).__init__(config, input_blob)

        # Create directory if it doesn't exist
        directory = os.path.join(settings.persisters.directory, '{:%Y_%m_%d}'.format(datetime.datetime.now()),
                                 settings.observation.name)
        filename = settings.observation.name + '_raw'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create file
        if config.use_timestamp:
            file_path = os.path.join(directory, "%s_%s" % (filename, str(time.time())))
        else:
            file_path = os.path.join(directory, filename + '.dat')

        self._base_filepath = file_path.split('.')[0]
        self._current_filepath = file_path
        self._chunk_size = 4  # number of data blobs to save in a raw file
        self._raw_file_counter = 1

        # Open file (if file exists, remove first)
        if os.path.exists(file_path):
            os.remove(file_path)

        self._file = open(file_path, "wb+")
        # fadvise.set_advice(self._file, fadvise.POSIX_FADV_SEQUENTIAL)

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

        self._file = self.open_file()

        # Copy input data to output data
        if output_data is not None:
            output_data[:] = input_data

        # Expand complex data, transpose and save to file
        data = input_data[0, 0, :, :]
        data = np.array([data.real, data.imag])
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))
        data = data.ravel()

        self._file.write(struct.pack('f' * len(data), *data))
        self._file.flush()

    def open_file(self):
        """
        Get the file path for the fits file

        :return:
        """

        if self._iter_count % self._chunk_size == 0:
            self._file.close()
            logging.info("Closed raw data file at {}".format(self._current_filepath))

            self._current_filepath = '{}_{}.dat'.format(self._base_filepath, self._raw_file_counter)

            self._file = open(self._current_filepath, "wb+")
            # fadvise.set_advice(self._file, fadvise.POSIX_FADV_SEQUENTIAL)

            logging.info("Opened a new raw file at {}".format(self._current_filepath))

            self._raw_file_counter += 1

        return self._file
