import datetime
import os
import pickle
import struct
import time

# import fadvise
import numpy as np

from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob


class BeamPersister(ProcessingModule):

    def __init__(self, config, input_blob=None):
        # Call superclass initialiser
        super(BeamPersister, self).__init__(config, input_blob)

        # Sanity checks on configuration
        if {'use_timestamp'} - set(config.settings()) != set():
            raise PipelineError("Persister: Missing keys on configuration. (use_timestamp)")

        # Create directory if it doesn't exist
        directory = os.path.join(settings.persisters.directory, '{:%Y_%m_%d}'.format(datetime.datetime.now()),
                                 settings.observation.name)
        filename = settings.observation.name + '_beam'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create file
        if config.use_timestamp:
            file_path = os.path.join(directory, "%s_%s" % (filename, str(time.time())))
        else:
            file_path = os.path.join(directory, filename + '.dat')

        # Open file (if file exists, remove first)
        if os.path.exists(file_path):
            os.remove(file_path)

        self._file = open(file_path, "wb+")        

        # Use fadvise to optimise 
        # fadvise.set_advice(self._file, fadvise.POSIX_FADV_SEQUENTIAL)

        # Initialise ranges to persist for file
        self._beam_range = slice(None)
        self._channel_range = slice(None)

        if 'channel_range' in self._config.settings():
            if type(self._config.channel_range) is list:
                self._channel_range = slice(self._config.channel_range[0], self._config.channel_range[1] + 1)
            else:
                self._channel_range = self._config.channel_range

        if 'beam_range' in self._config.settings():
            if type(self._config.beam_range) is list:
                self._beam_range = slice(self._config.beam_range[0], self._config.beam_range[1] + 1)
            else:
                self._beam_range = self._config.beam_range

        # Variable to check whether meta file has been written
        self._head_filepath = file_path + '.pkl'
        self._head_written = False

        # Counter
        self._counter = 0

        # Processing module name
        self.name = "Persister"

    def _get_filepath(self, config):
        # Create directory if it doesn't exist
        directory = os.path.join(settings.persisters.directory, '{:%Y_%m_%d}'.format(datetime.datetime.now()),
                                 settings.observation.name)
        filename = settings.observation.name + '_beam'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create file
        if config.use_timestamp:
            return os.path.join(directory, "%s_%s" % (filename, str(time.time())))
        else:
            return os.path.join(directory, filename)

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return ChannelisedBlob(self._config, self._input.shape, datatype=np.complex64)

    def process(self, obs_info, input_data, output_data):
        """

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # If head file not written, write it now
        if not self._head_written:
            obs_info['transmitter_frequency'] = settings.observation.transmitter_frequency
            obs_info['start_beam_in_file'] = self._beam_range.start if self._beam_range.start is not None else 0

            obs_info['nof_beams_in_file'] = obs_info['nbeams'] if self._beam_range.start is None else \
                self._beam_range.stop_module - self._beam_range.start

            obs_info['start_channel_in_file'] = self._channel_range.start if self._channel_range.start is not None else 0

            obs_info['nof_channels_in_file'] = obs_info['nchans'] if self._channel_range.start is None else \
                self._channel_range.stop_module - self._channel_range.start

            del obs_info['nsubs']

            with open(self._head_filepath, 'wb') as f:
                pickle.dump(obs_info.get_dict(), f)
            self._head_written = True

        # Save data to output
        output_data[:] = input_data[:].copy()

        # Ignore first 2 buffers (because of channeliser)
        self._counter += 1
        if self._counter <= 2:
            return obs_info

        # Transpose data and write to file
        if input_data.dtype == float:
            temp_array = input_data[self._beam_range, self._channel_range, :].T.ravel()
        elif 'compute_power' in settings.persister.__dict__.keys() and settings.persister.compute_power:
            temp_array = np.power(np.abs(input_data[self._beam_range, self._channel_range, :].T), 2).ravel()
        else:
            temp_array = input_data[self._beam_range, self._channel_range, :].T.ravel()
            temp_array = np.stack((temp_array.real, temp_array.imag), axis=1).ravel()

        self._file.write(struct.pack('f' * len(temp_array), *temp_array))
        self._file.flush()

        return obs_info

    def _get_filepath(self, obs_info):
        """
        Return the file path of the persisted data

        :param obs_info:
        :return:
        """

        directory = os.path.abspath(os.path.join(settings.persisters.directory, settings.observation.name,
                                                 '{:%Y-%m-%dT%H%M}'.format(obs_info['timestamp'])))
        filename = settings.observation.name + '_beam'

        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        return os.path.join(directory, filename)
