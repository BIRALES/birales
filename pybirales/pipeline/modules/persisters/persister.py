import logging
import os
import pickle
import time
import struct
from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.pipeline.base.processing_module import ProcessingModule
import numpy as np
import fadvise


class Persister(ProcessingModule):
    """ Dummy data generator """

    def __init__(self, config, input_blob=None):

        # Call superclass initialiser
        super(Persister, self).__init__(config, input_blob)

        # Sanity checks on configuration
        if {'directory'} - set(config.settings()) != set():
            raise PipelineError("Persister: Missing keys on configuration. (directory)")

        # Create directory if it doesn't exist
        if not os.path.exists(config.directory):
            os.makedirs(config.directory)

        # Create file
        if config.use_timestamp:
            filepath = os.path.join(config.directory, "%s_%s" % (config.filename, str(time.time())))
        else:
            if 'filename' not in config.settings():
                raise PipelineError("Persister: filename required when not using timestamp")
            filepath = os.path.join(config.directory, config.filename + '.dat')

        # Open file (if file exists, remove first)
        if os.path.exists(filepath):
            os.remove(filepath)

        self._file = open(filepath, "wb+")

        # Use fadvise to optimise 
        fadvise.set_advice(self._file, fadvise.POSIX_FADV_SEQUENTIAL)

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
        self._head_filepath = filepath + '.pkl'
        self._head_written = False

        # Counter
        self._counter = 0

        # Processing module name
        self.name = "Persister"

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return ChannelisedBlob(self._config, self._input.shape,
                         datatype=np.complex64)

    def process(self, obs_info, input_data, output_data):

        # If head file not written, write it now
        if not self._head_written:
            obs_info['transmitter_frequency'] = settings.observation.transmitter_frequency
            obs_info['start_beam_in_file'] = self._beam_range.start if self._beam_range.start is not None else 0

            obs_info['nof_beams_in_file'] = obs_info['nbeams'] if self._beam_range.start is None else \
                self._beam_range.stop - self._beam_range.start

            obs_info['start_channel_in_file'] = self._channel_range.start if self._channel_range.start is not None else 0

            obs_info['nof_channels_in_file'] = obs_info['nchans'] if self._channel_range.start is None else \
                self._channel_range.stop - self._channel_range.start

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
        # np.save(self._file, np.abs(input_data[self._beam_range, self._channel_range, :].T))
        temp_array = np.power(np.abs(input_data[self._beam_range, self._channel_range, :].T), 2).ravel()
        self._file.write(struct.pack('f' * len(temp_array), *temp_array))
        self._file.flush()

        return obs_info
