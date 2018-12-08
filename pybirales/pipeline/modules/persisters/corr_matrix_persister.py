import logging as log
import os
import pickle

import h5py
import numpy as np

from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
import datetime


def create_corr_matrix_filepath(timestamp):
    """
    Return the file path of the persisted data

    :param timestamp:
    :return:
    """
    root_dir = settings.persisters.directory
    # if settings.observation.type == 'calibration':
    #     root_dir = os.path.join(os.environ['HOME'], settings.calibration.tmp_dir)

    # Create directory if it doesn't exist
    directory = os.path.join(root_dir, '{:%Y_%m_%d}'.format(datetime.datetime.now()),
                             settings.observation.name)
    filename = '{}_{}.h5'.format(settings.observation.name, settings.corrmatrixpersister.filename_suffix)

    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    return os.path.join(directory, filename)


class CorrMatrixPersister(ProcessingModule):
    def __init__(self, config, input_blob=None):
        """

        :param config:
        :param input_blob:
        :return:
        """

        # Call superclass initializer
        super(CorrMatrixPersister, self).__init__(config, input_blob)

        # Sanity checks on configuration
        if {'filename_suffix', 'use_timestamp'} - set(config.settings()) != set():
            raise PipelineError("Persister: Missing keys on configuration. (filename_suffix, use_timestamp)")

        # Get the destination file path of the persisted data
        self._filepath = None

        # Variable to check whether meta file has been written
        self._head_filepath = None

        self._head_written = False

        # Counter
        self._counter = 0
        self._iterator = 0

        # Processing module name
        self.name = "CorrMatrixPersister"

    def generate_output_blob(self):
        """
        Generate output data blob

        :return:
        """

        return None

    @staticmethod
    def _create_pkl_file(filepath, obs_info):
        """

        :param filepath:
        :param obs_info:
        :return:
        """

        # Write observation information
        with open(filepath, 'wb') as f:
            pickle.dump(obs_info.get_dict(), f)

    @staticmethod
    def _create_hdf5_file(filepath, obs_info):
        """
        Create HDF5 file for storing correlation matrix

        :param filepath:
        :param obs_info:
        :return:
        """

        f = h5py.File(filepath, "w")

        dset = f.create_dataset("Vis", (obs_info['nsamp'], obs_info['nsubs'], obs_info['nbaselines'],
                                        obs_info['nstokes']),
                                maxshape=(None, obs_info['nsubs'], obs_info['nbaselines'], obs_info['nstokes']),
                                dtype='c16')

        dset[:] = np.zeros(((obs_info['nsamp'], obs_info['nsubs'], obs_info['nbaselines'],
                             obs_info['nstokes'])), dtype=np.complex64)

        # Create baselines data set
        dset2 = f.create_dataset("Baselines", (obs_info['nbaselines'], 3))

        antenna1, antenna2, baselines, counter = [], [], [], 0
        for i in range(obs_info['nants']):
            for j in range(i + 1, obs_info['nants']):
                antenna1.append(i)
                antenna2.append(j)
                baselines.append(counter)
                counter += 1

        dset2[:, :] = np.transpose([baselines, antenna1, antenna2])

        # Ready file
        f.flush()
        f.close()

    def process(self, obs_info, input_data, output_data):
        """
        Save correlation matrix to file

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # If first time running, create and initialise file
        if self._counter == 0:
            # Write the observation data file
            self._filepath = create_corr_matrix_filepath(obs_info['timestamp'])
            self._create_hdf5_file(self._filepath, obs_info)

            # Write header file
            self._head_filepath = self._filepath + '.pkl'
            self._create_pkl_file(self._head_filepath, obs_info)

            log.debug('Writing observation PKL and H5 file at {}'.format(os.path.abspath(self._filepath)))

        # Open file
        f = h5py.File(self._filepath, "a")

        # Write data to file
        size = self._counter * obs_info['nsamp']
        time_start, time_end = size, size + obs_info['nsamp']

        dset = f["Vis"]

        # Resize dataset if required
        if size >= len(dset[:, 0, 0, 0]):
            dset.resize(size * 2, axis=0)

        dset[range(time_start, time_end), :, :, :] = input_data[:, :, :, :]

        # Flush and close file
        f.flush()
        f.close()

        self._counter += 1

        return obs_info
