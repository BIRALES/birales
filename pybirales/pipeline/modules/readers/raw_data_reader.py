import datetime
import json
import logging as log
import pickle
import sys
import time

import numpy as np

from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError, ObservationInfo, NoDataReaderException, \
    BIRALESObservationException
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.dummy_data import DummyBlob
from pybirales.repository.message_broker import broker


class RawDataReader(ProcessingModule):
    """ Raw data reader """

    def __init__(self, config, input_blob=None):
        """

        :param config:
        :param input_blob:
        :return:
        """

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
        self._metrics_poll_freq = 10
        self._metric_channel = 'antenna_metrics'

        # Call superclass initialiser
        super(RawDataReader, self).__init__(config, input_blob)

        # Load the data file
        try:
            self._f = open(self._filepath, 'rb')
            # self._f.seek(self._nsamp * self._nants * 8 * 1000)
            log.info('Using raw data in: {}'.format(self._filepath))
        except IOError:
            log.error('Data not found in %s. Exiting.', self._filepath)
            raise BIRALESObservationException("Data not found in {}".format(self._filepath))

        # Load the PKL file
        try:
            self._config = pickle.load(open(self._filepath + config.config_ext, 'rb'))

            # Use the declination that is in the PKL file
            settings.beamformer.reference_declination = self._config['settings']['beamformer']['reference_declination']
        except IOError:
            log.error('Config PKL file was not found in %s. Exiting.', self._filepath + config.config_ext)
            raise BIRALESObservationException("Config PKL file was not found")

        # Processing module name
        self.name = "RawDataReader"

    @staticmethod
    def _calculate_rms(input_data):
        """
        Calculate the RMS of the incoming antenna data
        :param input_data: Input antenna data
        :return:
        """

        return np.squeeze(np.sqrt(np.sum(np.power(np.abs(input_data), 2.), axis=2)))

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
        """

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # if self._read_count == 20:
        #     raise BIRALESObservationException("Observation finished")

        data = self._f.read(self._nsamp * self._nants * 8)

        if data is "":
            log.warning('End of file reached. Raising "No Data" exception.')
            raise NoDataReaderException

        data = np.frombuffer(data, np.complex64)

        try:
            data = data.reshape((1, 1, self._nsamp, self._nants))
        except ValueError:
            # Sleep the thread before calling a no data - wait for the other modules to finish
            # todo - this could be handled better
            # time.sleep(20)

            raise BIRALESObservationException("Observation finished")

        output_data[:] = data

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
        obs_info['timestamp'] = self._config['timestamp'] + datetime.timedelta(
            seconds=self._nsamp * obs_info['sampling_time']) * self._read_count

        self.publish_antenna_metrics(data, obs_info)

        self._read_count += 1

        return obs_info

    def publish_antenna_metrics(self, data, obs_info):
        if self._read_count % self._metrics_poll_freq == 0:
            rms_voltages = self._calculate_rms(data).tolist()
            timestamp = obs_info['timestamp'].isoformat('T')

            msg = json.dumps({'timestamp': timestamp,
                              'voltages': rms_voltages})
            broker.publish(self._metric_channel, msg)

            log.debug('Published antenna metrics %s: %s', timestamp,
                      ', '.join(['%0.2f'] * len(rms_voltages)) % tuple(rms_voltages))