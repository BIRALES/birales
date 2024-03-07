import datetime
import logging
import os
import pickle
import struct
import time

import h5py
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
        dir_path = os.path.expanduser(settings.persisters.directory)
        directory = os.path.join(dir_path, '{:%Y_%m_%d}'.format(datetime.datetime.now()),
                                 settings.observation.name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create file
        self._raw_file = os.path.join(directory, f"{settings.observation.name}_raw.h5")

        # Open file (if file exists, remove first)
        if os.path.exists(self._raw_file):
            os.remove(self._raw_file)

        # Iteration counter
        self._counter = 0

        # Processing module name
        self.name = "RawPersister"

    def _create_output_file(self, obs_info):
        """ Create the HDF5 output file, including metadata """
        with h5py.File(self._raw_file, 'w') as f:
            # Create group that will contain observation information
            info = f.create_group('observation_information')

            # ADd attributes to group
            info.attrs['observation_name'] = settings.observation.name
            info.attrs['transmitter_frequency'] = obs_info['transmitter_frequency']
            info.attrs['sampling_time'] = obs_info['sampling_time']
            info.attrs['start_center_frequency'] = obs_info['start_center_frequency']
            info.attrs['channel_bandwidth'] = obs_info['channel_bandwidth']
            info.attrs['reference_declinations'] = obs_info['declinations']
            info.attrs['observation_settings'] = str(obs_info['settings'])

            # TODO: Add number of antennas
            info.attrs['nof_antennas'] = 0

            # TODO: Add calibration coefficients

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return ReceiverBlob(self._input.shape, datatype=np.complex64)

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

            logging.info("Opened a new raw file at {}".format(self._current_filepath))

            self._raw_file_counter += 1

        return self._file
