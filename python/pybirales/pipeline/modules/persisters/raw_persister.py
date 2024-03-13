import copy
import math
from collections import deque
import datetime
import os

import h5py
import numpy as np

from pybirales import settings
from pybirales.pipeline.base.definitions import TriggerException
from pybirales.pipeline.base.processing_module import ProcessingModule


class RawPersister(ProcessingModule):
    """ Raw data persister """

    def __init__(self, config, input_blob=None):
        # Call superclass initialiser
        super(RawPersister, self).__init__(config, input_blob)

        # Create directory if it doesn't exist
        dir_path = os.path.expanduser(settings.persisters.directory)
        self._output_directory = os.path.join(dir_path, '{:%Y_%m_%d}'.format(datetime.datetime.now()),
                                              settings.observation.name)
        if not os.path.exists(self._output_directory):
            os.makedirs(self._output_directory)

        # Output file handle
        self._output_filename = None

        # If triggering is enabled, create the queue that will act as the buffer to store
        # historical data
        self._trigger_buffer = None
        self._trigger_timestamps = None
        self._trigger_block_duration = None

        # If triggering is enabled, use a separate thread to listen for request to process triggers
        self._trigger_thread = None

        # Latest obs_info
        self._latest_obs_info = None

        # Processing module name
        self.name = "RawPersister"

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return None

    def _create_output_file(self, obs_info, filename=None):
        """ Create the HDF5 output file, including metadata """

        # If the filename is None, generate one using the observation name
        if filename is None:
            self._output_filename = os.path.join(self._output_directory, f"{settings.observation.name}_raw.h5")

        # If the file already exists, remove it
        if os.path.exists(self._output_filename):
            os.remove(self._output_filename)

        # Create H5 output file with required metadata
        with h5py.File(self._output_filename, 'w') as f:
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

            # Create group that will contain all observation data
            f.create_group('observation_data')

    def _add_raw_data_to_file(self, obs_info, raw_data, nof_samples=0, timestamp=None):
        """ Add raw data to output file """

        # If the provided number of samples is 0, then use the shape of the raw data
        if nof_samples == 0:
            nof_samples = raw_data.shape[0]

        # If the provided timestamp is None, then use the timestamp specified in obs_info
        if timestamp is None:
            timestamp = obs_info['timestamp']

        with h5py.File(self._output_filename, 'a') as f:
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
                                    chunks=(2 ** 14 / (obs_info['nof_antennas'] * np.dtype(np.complex64).itemsize),
                                            obs_info['nof_antennas']),
                                    dtype=np.complex64)

                dset.create_dataset(timestamp_name, 0, maxshape=(None,), dtype='f8')

            # Add data to file
            dset = f[f'observation_data/{dataset_name}']
            dset.resize((dset.shape[0] + nof_samples, dset.shape[1]))
            dset[-nof_samples:, :] = raw_data

            # Add timestamp to file
            dset = f[f"observation_data/{timestamp_name}"]
            dset.resize((dset.shape[0] + nof_samples,))
            timestamp = datetime.datetime.timestamp(timestamp)
            dset[-nof_samples:] = timestamp + np.arange(nof_samples) * obs_info['sampling_time']

    def _initialise_trigger_buffer(self, obs_info):
        """ Initialise the trigger buffer for storing historical data """
        block_duration = settings.observation.samples_per_second / obs_info['nof_samples']
        buffer_length = settings.manager.trigger_buffer_length + settings.manager.trigger_time_delta * 2
        buffer_length = math.ceil(buffer_length * block_duration)
        self._trigger_buffer = deque(maxlen=buffer_length)
        self._trigger_timestamps = deque(maxlen=buffer_length)
        self._trigger_block_duration = block_duration

    def _add_blob_to_trigger_buffer(self, obs_info, input_data):
        """ Add the current input_data to the trigger buffer, as well as additional metadata"""
        self._trigger_buffer.appendleft(input_data.copy())
        self._trigger_timestamps.appendleft(obs_info['timestamp'])
        self._latest_obs_info = copy.copy(obs_info)

    def trigger_to_file(self, start_time, duration, identifier="trigger"):
        """ A trigger was generated to write parts of the trigger buffer to file. Find the corresponding
            segment in the buffer and write it to file """

        # TODO: Define error handling mechanism

        # Determine whether required trigger start and duration falls within current buffer
        if len(self._trigger_buffer) == 0:
            raise TriggerException("Trigger buffer is empty")

        if (self._trigger_timestamps[-1] < start_time or
                self._trigger_timestamps[0] - self._trigger_buffer[-1] < duration):
            raise TriggerException("Requested trigger time window is not wholly available in trigger buffer")

        # Requested trigger window is in file. Creat trigger file
        self._create_output_file(self._latest_obs_info, filename=f"{identifier}_raw.h5")

        # Some pre-computations
        stop_time = start_time + duration
        block_duration = settings.observation.samples_per_second / self._latest_obs_info['nof_samples']

        # Find the start and end blobs within the trigger, as well as the bounds within each blob
        first_blob = np.argwhere((np.array(self._trigger_timestamps) + block_duration) - start_time > 0)[0]
        first_blob_shift = math.floor((start_time - self._trigger_timestamps[first_blob]) *
                                      settings.observation.samples_per_second)

        last_blob = np.argwhere((np.array(self._trigger_timestamps) + block_duration) - stop_time > 0)[0]
        last_blob_shift = math.ceil((stop_time - self._trigger_timestamps[last_blob]) *
                                    settings.observation.samples_per_second)

        # Write first blob
        self._add_raw_data_to_file(self._latest_obs_info,
                                   self._trigger_buffer[first_blob][first_blob_shift:],
                                   self._latest_obs_info['nof_samples'] - first_blob_shift,
                                   self._trigger_timestamps[first_blob] + first_blob_shift *
                                   self._latest_obs_info['sampling_time'])

        # Write all intermediary blobs
        for blob_number in range(first_blob + 1, last_blob):
            self._add_raw_data_to_file(self._latest_obs_info,
                                       self._trigger_buffer[blob_number])

        # Write last blob
        self._add_blob_to_trigger_buffer(self._latest_obs_info, self._trigger_buffer[last_blob][:last_blob_shift])

        # Remove file handle
        self._output_filename = None

    def wait_for_trigger(self):
        """ Runs in a separate thread to wait for trigger requests """
        pass

    def process(self, obs_info, input_data, output_data):
        """ Write received raw data to output file """

        # If triggering is enabled, do not write directly to file but simply add to the trigger buffer
        if settings.manager.detection_trigger_enabled:
            if self._trigger_buffer is None:
                self._initialise_trigger_buffer(obs_info)
            else:
                self._add_blob_to_trigger_buffer(obs_info, input_data)
            return obs_info

        # If this is the first iteration, create the output file
        if self._output_filename is None:
            self._create_output_file(obs_info)

        # ADd raw data to output file (ignore polarizations and sub-bands for now
        self._add_raw_data_to_file(obs_info, input_data[0, 0, :])

        return obs_info

