import numpy as np
import pickle
import os
import time
import h5py

from pybirales.base.definitions import PipelineError
from pybirales.base.processing_module import ProcessingModule


class CorrMatrixPersister(ProcessingModule):
    """ Dummy data generator """

    def __init__(self, config, input_blob=None):

        # Call superclass initialiser
        super(CorrMatrixPersister, self).__init__(config, input_blob)

        # Sanity checks on configuration
        if {'directory', 'filename', 'use_timestamp'} - set(config.settings()) != set():
            raise PipelineError("Persister: Missing keys on configuration. (directory, filename, use_timestamp)")

        # Create directory if it doesn't exist
        if not os.path.exists(config.directory):
            os.mkdir(config.directory)

        # Create file
        if config.use_timestamp:
            self._filepath = os.path.join(config.directory, "%s_%s" % (config.filename, str(time.now())))
        else:
            if 'filename' not in config.settings():
                raise PipelineError("CorrMatrixPersister: filename required when not using timestamp")
            self._filepath = os.path.join(config.directory, config.filename + '.h5')

        # Variable to check whether meta file has been written
        self._head_filepath = self._filepath + '.pkl'
        self._head_written = False

        # Counter
        self._counter = 0
        self._iterator = 0

        # Processing module name
        self.name = "CorrMatrixPersister"

    def generate_output_blob(self):
        """ Generate output data blob """
        return None

    def _create_hdf5_file(self, obs_info):
        """ Create HDF5 file for storing correlation matrix"""
        f = h5py.File(self._filepath, "w")

        dset = f.create_dataset("Vis", (64, obs_info['nsubs'], obs_info['nbaselines'],
                                        obs_info['nstokes']),
                                maxshape=(None, obs_info['nsubs'], obs_info['nbaselines'], obs_info['nstokes']),
                                dtype='c16')

        dset[:] = np.zeros(((64, obs_info['nsubs'], obs_info['nbaselines'],
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
        """ Save correlation matrix to file """

        # If first time running, create and initialise file
        if self._counter == 0:
            self._create_hdf5_file(obs_info)

            # Write observation information
            with open(self._head_filepath, 'wb') as f:
                pickle.dump(obs_info.get_dict(), f)

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