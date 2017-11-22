import glob
import logging
import os

import numpy as np
from pydaq.persisters.channel import ChannelFormatFileManager
from pydaq.persisters.definitions import FileModes

from pybirales.pipeline.base.definitions import PipelineError, ObservationInfo
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.receiver_data import ReceiverBlob
from pybirales import settings


class AAVSChannelReader(ProcessingModule):
    """ Receiver """

    def __init__(self, config, input_blob=None):

        # This module does not need an input block
        if input_blob is not None:
            raise PipelineError("AAVSChannelReader: Receiver does not need an input block")

        # Sanity checks on configuration
        if {'nsamp', 'nants', 'nsubs', 'npols'} - set(config.settings()) != set():
            raise PipelineError("AAVSChannelReader: Missing keys on configuration "
                                "(nsamp, nants, nsubs, npol)")

        self._nsamp = config.nsamp
        self._nants = config.nants
        self._nsubs = config.nsubs
        self._npols = config.npols
        self._directory = config.directory

        # Remove any pending lock file
        for f in glob.glob(os.path.join(config.directory, "*.lock")):
            os.remove(f)

        # Ge channel file handler
        self._file_manager = ChannelFormatFileManager(root_path='/home/lessju/Desktop', mode=FileModes.Read)

        self._current_offset = 0

        # Call superclass initialiser
        super(AAVSChannelReader, self).__init__(config, input_blob)

        # Processing module name
        self.name = "AAVSChannelReader"

    def generate_output_blob(self):
        """ Generate output data blob """
        # Generate blob
        return ReceiverBlob(self._config, [('npols', self._npols),
                                           ('nsubs', self._nsubs),
                                           ('nsamp', self._nsamp),
                                           ('nants', self._nants)],
                            datatype=np.complex64)

    def process(self, obs_info, input_data, output_data):
        """ Read data from file """
        #
        output, timestamps = self._file_manager.read_data(channels=[0], antennas=[i for i in range(self._nants)],
                                                          polarizations=[i for i in range(self._npols)],
                                                          n_samples=self._nsamp,
                                                          sample_offset=self._current_offset,
                                                          date_time=None)

        output = (output['real'] + 1j * output['imag']).astype(np.complex64)
        output = np.transpose(output, (1, 0, 2, 3))
        output = np.transpose(output, (2, 1, 0, 3))
        output = np.transpose(output, (0, 1, 3, 2))
        output_data[:] = output

        # Update offset
        self._current_offset += self._nsamp

        # Create observation information
        obs_info = ObservationInfo()
        obs_info['nsubs'] = self._nsubs
        obs_info['nsamp'] = self._nsamp
        obs_info['nants'] = self._nants
        obs_info['npols'] = self._npols
        obs_info['sampling_time'] = 1.0 / settings.observation.samples_per_second
        obs_info['start_center_frequency'] = settings.observation.start_center_frequency
        obs_info['channel_bandwidth'] = settings.observation.channel_bandwidth
        obs_info['timestamp'] = timestamps[0]

        logging.info("Read channelized data")

        return obs_info
