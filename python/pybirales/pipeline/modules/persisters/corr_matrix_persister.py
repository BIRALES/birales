import datetime
import logging as log
import os
import pickle

import h5py
import numpy as np
import pytz

from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob


def create_corr_matrix_filepath():
    """
    Return the file path of the persisted data

    :return:
    """

    if settings.corrmatrixpersister.corr_matrix_filepath:
        corr_matrix_filepath = settings.corrmatrixpersister.corr_matrix_filepath
    else:
        now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        dir_path = os.path.expanduser(settings.persisters.directory)
        basedir = os.path.join(dir_path, '{:%Y_%m_%d}'.format(now),
                               settings.observation.name)
        corr_matrix_filepath = os.path.join(basedir, settings.observation.name + '__corr.h5')

        if not os.path.exists(basedir):
            os.makedirs(basedir)

    return corr_matrix_filepath


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
        if {'use_timestamp'} - set(config.settings()) != set():
            raise PipelineError("Persister: Missing keys on configuration. (use_timestamp)")

        # Get the destination file path of the persisted data
        self._filepath = None

        # Variable to check whether meta file has been written
        self._head_filepath = None

        self._head_written = False

        # Counter
        self._counter = 0
        self._iterator = 0

        self._after_channeliser = True if type(input_blob) is ChannelisedBlob else False

        self._nof_channels = None

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

    def _create_hdf5_file(self, filepath, obs_info):
        """
        Create HDF5 file for storing correlation matrix

        :param filepath:
        :param obs_info:
        :return:
        """

        if not self._nof_channels:
            self._nof_channels = self._get_nof_channels(obs_info, self._after_channeliser)

        f = h5py.File(filepath, "w")

        dset = f.create_dataset("Vis", (obs_info['nof_samples'], self._nof_channels, obs_info['nof_baselines'], obs_info['nof_stokes']),
                                maxshape=(None, self._nof_channels, obs_info['nof_baselines'], obs_info['nof_stokes']),
                                dtype='c16')

        dset[:] = np.zeros(((obs_info['nof_samples'], self._nof_channels, obs_info['nof_baselines'], obs_info['nof_stokes'])),
                           dtype=np.complex64)

        # Create baselines data set
        dset2 = f.create_dataset("Baselines", (obs_info['nof_baselines'], 3))

        antenna1, antenna2, baselines, counter = [], [], [], 0
        for i in range(obs_info['nof_antennas']):
            for j in range(i + 1, obs_info['nof_antennas']):
                antenna1.append(i)
                antenna2.append(j)
                baselines.append(counter)
                counter += 1

        dset2[:, :] = np.transpose([baselines, antenna1, antenna2])

        # Create timesample dataset

        dset_time = f.create_dataset("Time", shape=(obs_info['nof_samples'],), maxshape=(None,), dtype=float, chunks=True)
        dset_time[:] = np.zeros((obs_info['nof_samples']), dtype=float)

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
            self._filepath = create_corr_matrix_filepath()
            self._create_hdf5_file(self._filepath, obs_info)

            # Write header file
            self._head_filepath = self._filepath + '.pkl'
            self._create_pkl_file(self._head_filepath, obs_info)

            log.debug('Writing observation PKL and H5 file at {}'.format(os.path.abspath(self._filepath)))

        # Open file
        f = h5py.File(self._filepath, "a")

        # Write data to file
        size = self._counter * obs_info['nof_samples']
        time_start, time_end = size, size + obs_info['nof_samples']

        dset = f["Vis"]

        # Resize dataset if required
        if size >= len(dset[:, 0, 0, 0]):
            dset.resize(size * 2, axis=0)
            f["Time"].resize(size * 2, axis=0)

        dset[time_start:time_end, :, :, :] = input_data[:, :, :, :]

        timestamps = datetime.timedelta(seconds=obs_info['sampling_time']) * np.arange(obs_info['nof_samples']) + obs_info[
            'timestamp']
        f["Time"][time_start:time_end] = [t.timestamp() for t in timestamps]

        # Flush and close file
        f.flush()
        f.close()

        self._counter += 1

        return obs_info

    def _get_nof_channels(self, obs_info, after_channeliser):
        nof_channels = obs_info['nof_subbands']
        if after_channeliser:
            nof_channels = obs_info['nof_channels']

            if hasattr(settings.correlator, 'channel_start') and hasattr(settings.correlator, 'channel_end'):
                nof_channels = settings.correlator.channel_end - settings.correlator.channel_start

        return nof_channels
