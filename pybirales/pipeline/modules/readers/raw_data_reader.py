import sys
import numpy as np
from time import sleep
import datetime

from pybirales.pipeline.base.definitions import PipelineError, ObservationInfo, NoDataReaderException
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.dummy_data import DummyBlob
from pybirales import settings
import logging as log
import pickle
import time


class RawDataReader(ProcessingModule):
    """ Raw data reader """

    def __init__(self, config, input_blob=None):

        # This module does not need an input_blob
        self._validate_data_blob(input_blob, valid_blobs=[type(None)])

        # Sanity checks on configuration
        if {'nants', 'nsamp', 'nsubs', 'npols'} - set(config.settings()) != set():
            raise PipelineError("DummyDataGenerator: Missing keys on configuration "
                                "(nants, nsamp, nsub, 'npols')")
        self._nants = config.nants
        self._nsamp = config.nsamp
        self._nsubs = config.nsubs
        self._npols = config.npols
        self._filepath = config.filepath
        self._read_count = 0

        # Call superclass initialiser
        super(RawDataReader, self).__init__(config, input_blob)

        # Open file
        try:
            self._f = open(self._filepath, 'rb')
        except IOError:
            log.error('Data not found in %s. Exiting.', self._filepath)
            sys.exit()

        try:
            self._config = pickle.load(open(config.config_filepath, 'rb'))
        except IOError:
            log.error('Config PKL file was not found in %s. Exiting.', config.config_filepath)
            sys.exit()

        # Processing module name
        self.name = "RawDataReader"

    def generate_output_blob(self):
        """
        Generate the output blob
        :return:
        """
        return DummyBlob(self._config, [('npols', self._npols),
                                        ('nsubs', self._nsubs),
                                        ('nsamp', self._nsamp),
                                        ('nants', self._nants)],
                         datatype=np.complex64)

    def process(self, obs_info, input_data, output_data):
        # Read next data set

        data = self._f.read(self._nsamp * self._nants * 8)
        data = np.frombuffer(data, np.complex64)
        try:
            data = data.reshape((1, 1, self._nsamp, self._nants))
        except ValueError:
            # Sleep the thread before calling a no data - wait for the other modules to finish
            # todo - this could be handled better
            time.sleep(20)
            raise NoDataReaderException

        output_data[:] = data
        sleep(0.5)

        # output_data = self.generate_corrdata()
        # Create observation information
        obs_info = ObservationInfo()
        obs_info['sampling_time'] = 1. / settings.observation.samples_per_second
        obs_info['nsubs'] = self._nsubs
        obs_info['nsamp'] = self._nsamp
        obs_info['nants'] = self._nants
        obs_info['npols'] = self._npols

        obs_info['transmitter_frequency'] = self._config['settings']['observation']['transmitter_frequency']
        obs_info['start_center_frequency'] = self._config['start_center_frequency']
        obs_info['channel_bandwidth'] = settings.observation.channel_bandwidth
        obs_info['timestamp'] = self._config['timestamp'] + datetime.timedelta(seconds=self._nsamp * obs_info['sampling_time']) * self._read_count

        self._read_count += 1

        return obs_info
