import datetime
import json
import logging as log
import os
import pickle

import numpy as np

from pybirales import settings
from pybirales.pipeline.base.definitions import PipelineError, ObservationInfo, BIRALESObservationException
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

        self._read_count = 48  # norad 1328 on 03/05/2019
        self._read_count = 53  # norad 41182 on 03/05/2019 @ 11
        self._read_count = 35  # norad 20666 on 11/02/2019 @ 11
        self._read_count = 60  # norad 25160 on 03/10/2019 @ 06
        self._read_count = 50  # norad 40894 on 11/02/2019 @ 10:53
        self._read_count = 70  # norad 41240 on 11/02/2019 @ 11:36

        self._read_count = 20  # norad 4259 on 03/10/2019 @ 06:27

        self._read_count = 20  # norad 1328 on 03/10/2019 @ 06:27

        self._metrics_poll_freq = 10
        self._metric_channel = 'antenna_metrics'

        self._base_filepath = config.filepath.split('.')[0]

        self._raw_file_counter = 0

        # Call superclass initialiser
        super(RawDataReader, self).__init__(config, input_blob)

        # Load the PKL file
        try:
            self._config = pickle.load(open(self._filepath + config.config_ext, 'rb'))

            # Use the declination that is in the PKL file
            settings.beamformer.reference_declination = self._config['settings']['beamformer']['reference_declination']


        except IOError:
            log.error('Config PKL file was not found in %s. Exiting.', self._filepath + config.config_ext)
            raise BIRALESObservationException("Config PKL file was not found")

        self._raw_file_timerange_display(self._filepath, self._config['timestamp'])

        # Load the data file
        try:
            self._f = self._get_start_file(self._filepath, self._read_count)

            # self._f = open(self._filepath, 'rb')
            # self._f.seek(self._nsamp * self._nants * 8 * self._read_count)

            log.info('Using raw data in: {}'.format(self._filepath))
        except IOError:
            log.error('Data not found in %s. Exiting.', self._filepath)
            raise BIRALESObservationException("Data not found in {}".format(self._filepath))

        # Processing module name
        self.name = "RawDataReader"

    def _get_start_file(self, filepath, skip):
        self._f = open(filepath, 'rb')

        if os.stat(filepath).st_size < self._nsamp * self._nants * 8 * skip:
            next_file = '{}_{}.dat'.format(self._base_filepath, self._raw_file_counter + 1)
            self._raw_file_counter += 1

            log.info("%s was skipped. Blobs to skip: %d. Next file: %s", os.path.basename(filepath), skip, next_file)

            # self._raw_file_timerange(filepath, self._config['timestamp'])

            skip -= os.stat(filepath).st_size / (self._nsamp * self._nants * 8)

            return self._get_start_file(next_file, skip)

        self._f.seek(self._nsamp * self._nants * 8 * skip)

        log.info("RawDataReader will use: %s and skip %d blobs from it. (read counter: %s)", filepath, skip,
                 self._read_count)

        return self._f

    def _raw_file_timerange_display(self, filepath, t0):

        raw_file = filepath
        sampling_time = 1. / 78125

        td = datetime.timedelta(seconds=self._nsamp * sampling_time)
        n_blobs = 0
        _raw_file_counter = 0
        while os.path.exists(raw_file):
            c_blobs = n_blobs
            n_blobs += os.stat(raw_file).st_size / (self._nsamp * self._nants * 8)
            print  os.path.basename(raw_file), t0 + td * c_blobs, ' to ', t0 + td * n_blobs, (
                        n_blobs - c_blobs), 'blobs', td * (n_blobs - c_blobs)

            _raw_file_counter += 1
            raw_file = '{}_{}.dat'.format(self._base_filepath, _raw_file_counter)


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

    def _change_raw_file(self):
        # check if a new file exists:
        next_file = '{}_{}.dat'.format(self._base_filepath, self._raw_file_counter + 1)

        if os.path.exists(next_file):
            self._f.close()
            log.info('Stopped reading from raw file: {}'.format(self._filepath))

            self._filepath = next_file

            self._f = open(self._filepath, 'rb')
            log.info('Using raw data in: {}'.format(self._filepath))

            self._raw_file_counter += 1

            return self._f

        return None

    def process(self, obs_info, input_data, output_data):
        """

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        data = self._f.read(self._nsamp * self._nants * 8)

        # Check if we have a complete set of data
        if len(data) < self._nsamp * self._nants * 8:
            # Change if there is any raw data file left
            self._f = self._change_raw_file()

            if not self._f:
                log.info("Data finished successfully. Stopping modules at iteration %d", self._iter_count)
                obs_info['stop_pipeline_at'] = self._iter_count
                self.stop()

                return
                # time.sleep(200)
                # raise NoDataReaderException("Observation finished")

            # Read from the next set of data from new file
            data = self._f.read(self._nsamp * self._nants * 8)

        try:
            data = np.frombuffer(data, np.complex64)
            data = data.reshape((1, 1, self._nsamp, self._nants))
        except ValueError:
            raise BIRALESObservationException("An error has occurred whilst reading the raw data file")

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
