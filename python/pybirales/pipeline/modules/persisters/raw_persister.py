import math
from collections import deque
import datetime
import os

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

        # If triggering is enabled, create the queue that will act as the buffer to store
        # historical data
        self._trigger_buffer = None
        self._trigger_timestamps = None

        # Processing module name
        self.name = "RawPersister"

    def _initialise_trigger_buffer(self, obs_info):
        """ Initialise the trigger buffer for storing historical data """
        buffer_length = settings.manager.trigger_buffer_length + settings.manager.trigger_time_delta * 2
        buffer_length = math.ceil(buffer_length * settings.observation.samples_per_second / obs_info['nof_samples'])
        self._trigger_buffer = deque(maxlen=buffer_length)
        self._trigger_timestamps = deque(maxlen=buffer_length)

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
            info.attrs['nof_antennas'] = obs_info['nof_antennas']
            info.attrs['channel_bandwidth'] = obs_info['channel_bandwidth']
            info.attrs['reference_declinations'] = obs_info['declinations']
            info.attrs['observation_settings'] = str(obs_info['settings'])

            # TODO: Add calibration coefficients

    def _add_blob_to_trigger_buffer(self, obs_info, input_data):
        """ Add the current input_data to the trigger buffer"""
        self._trigger_buffer.append_left(input_data.copy())
        self._trigger_timestamps.append_left(input_data['timestamp'])

    def _add_raw_data_to_file(self, obs_info, raw_data):
        """ Add raw data to output file """
        with h5py.File(self._raw_file, 'a') as f:
            # Create data set name
            dataset_name = "raw_data"
            timestamp_name = "raw_timestamp"

            # Load observation data group
            dset = f['observation_data']

            # If datasets do not exist, add them
            if dataset_name not in dset.keys():
                dset.create_dataset(dataset_name,
                                    (0, obs_info['nof_antennas']),
                                    maxshape=(None, obs_info['nof_antennas']),
                                    chunks=(obs_info['nof_samples'], obs_info['nof_antennas']),
                                    dtype=np.complex64)

                dset.create_dataset(timestamp_name, 0, maxshape=(None,), dtype='f8')

            # Add data to file
            dset = f[f'observation_data/{dataset_name}']
            dset.resize((dset.shape[0] + obs_info['nof_samples'], dset.shape[1]))
            dset[-obs_info['nof_samples']:, :] = raw_data

            # Add timestamp to file
            dset = f[f"observation_data/{timestamp_name}"]
            dset.resize((dset.shape[0] + obs_info['nof_samples'], ))
            timestamp = datetime.datetime.timestamp(obs_info['timestamp'])
            dset[-obs_info['nof_samples']:] = timestamp + np.arange(obs_info['nof_samples']) * obs_info['sampling_time']

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return ReceiverBlob(self._input.shape, datatype=np.complex64)

    def process(self, obs_info, input_data, output_data):
        """ Write received raw data to output file """

        # If triggering is enabled, do not write directly to file but simply add to the trigger buffer
        if self._trigger_buffer is not None:
            self._add_blob_to_trigger_buffer(obs_info, input_data)
            return obs_info

        # If this is the first iteration, create the output file
        if self._counter == 0:
            self._create_output_file(obs_info)

        # ADd raw data to output file (ignore polarizations and subbands for now
        self._add_raw_data_to_file(obs_info, input_data[0, 0, :])

        return obs_info
