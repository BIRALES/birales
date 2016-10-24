import logging as log
import os
import pickle
import time
from datetime import datetime
import numpy as np
from beam import Beam
from configuration.application import config
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial


class DataSet:
    """
    The DataSet class encapsulates the logic for reading and creating the beam data from the data set
    that was generated by the PyBirales Backend
    """
    config_ext = '.dat.pkl'
    data_set_ext = '.dat'

    def __init__(self, observation_name, data_set_name, n_beams):
        """
        Initialise the Data Set object
        :param observation_name:
        :param data_set_name:
        :param n_beams:
        :return:
        """
        self.observation_name = observation_name
        self.name = data_set_name
        self.id = observation_name + '.' + self.name
        self.data_file_path = self._get_data_file_path(self.observation_name, data_set_name)
        self.config_file_path = self._get_config_file_path(self.observation_name, data_set_name)

        self.config = self._init_data_set_config(self.config_file_path)
        self.n_beams = int(n_beams) or self.config['nbeams']
        self.n_channels = self.config['nchans']
        self.tx = self.config['transmitter_frequency']

        log.info('Extracting beam data from data set %s', data_set_name)
        # todo - this can be part of a multi-beam class instead of being associated with the data set
        self.beams = self.get_beams()

    def _get_data_file_path(self, observation_name, data_set_name):
        base_path = config.get('io', 'DATA_FILE_PATH')
        return os.path.join(base_path, observation_name, data_set_name, data_set_name + self.data_set_ext)

    def _get_config_file_path(self, observation_name, data_set_name):
        base_path = config.get('io', 'DATA_FILE_PATH')
        return os.path.join(base_path, observation_name, data_set_name, data_set_name + self.config_ext)

    def _read_data_set(self, data_set_file_path, n_beams, n_channels):
        """
        Read beam data from the data_set file associated with this observation
        :param n_beams: The number of beams
        :param n_channels: The number of channels
        :return: The processed beam data
        """
        log.info('Reading data set at %s', data_set_file_path)
        start = time.time()
        if os.path.isfile(data_set_file_path):
            data = np.fromfile(data_set_file_path, dtype=np.dtype('f'))
            limit = n_beams / float(self.config['nbeams']) * len(data)
            data = data[:int(limit)]
            n_samples = len(data) / (n_beams * n_channels)
            data = np.reshape(data, (n_samples, n_channels, n_beams))

            log.info('Data set data loaded in %s seconds', DataSet._time_taken(start))
            log.info('Read %s samples (%s s), %s channels, %s beams', n_samples,
                     round(self.config['sampling_rate'] * n_samples, 2), n_channels, n_beams)

            return data

        raise IOError('Data set was not found at ' + data_set_file_path)

    def get_beams(self):
        """
        Create and return a list of Beam objects from the beam data.

        :return: A list of Beams extracted from the data set data
        """

        # Read the data set data
        data_set_data = self._read_data_set(self.data_file_path, self.config['nbeams'], self.config['nchans'])

        # Initialise thread pool
        pool = ThreadPool(16)

        # Pass the data set data to the create beam function
        create_beam_func = partial(self._create_beam, data_set_data)

        # Create the iterable
        n_beams = range(0, self.n_beams)

        # Collect the beam data processed by N threads
        beams = pool.map(create_beam_func, n_beams)

        pool.close()
        pool.join()

        return beams

    def _create_beam(self, data_set_data, n_beam):
        log.debug('Generating beam %s from data set %s', n_beam, self.name)
        beam = Beam(beam_id=n_beam,
                    dec=0.0,
                    ra=0.0,
                    ha=0.0,
                    top_frequency=0.0,
                    frequency_offset=0.0,
                    data_set=self, beam_data=data_set_data)
        return beam

    @staticmethod
    def _init_data_set_config(config_file_path):
        """
        Configure this observation with the settings of the pickle file in the data_set
        :param config_file_path: The file path of where the data set configuration is located
        :return:
        """

        if os.path.isfile(config_file_path):
            data_set_config = pickle.load(open(config_file_path, "rb"))
            # todo - change these or remove
            # todo - validate settings read from pickle
            data_set_config['n_sub_channels'] = data_set_config['nchans']
            data_set_config['sampling_rate'] = data_set_config['sampling_time']
            data_set_config['f_ch1'] = data_set_config['start_center_frequency']
            data_set_config['f_off'] = data_set_config['channel_bandwidth']

            return data_set_config

        raise IOError('Config file was not found at ' + config_file_path)

    @staticmethod
    def _time_taken(start):
        time_taken = time.time() - start
        return round(time_taken, 2)

    def __iter__(self):
        """
        Get a dict representation of this data set object
        todo - add last updated and created time stamps

        :return:
        """
        yield 'name', self.name
        yield 'observation', self.observation_name
        yield 'n_channels', self.n_channels
        yield 'n_beams', self.n_beams
        yield 'tx', self.tx
        yield 'created_at', datetime.now()
        yield 'config', self.config
