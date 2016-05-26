import os
import config.application as config
import numpy as np
import pickle
import logging as log

from xml.dom import minidom
from models.Beam import Beam


# todo configure observation with pickle file
# todo - Change config to a YAML

class Observation:
    """
    The Observation Model encodes the properties of an observation campaign. Multiple Beam objects are associated with
    each Observation object.
    """

    def __init__(self, name, data_set):
        """
        Initialisation of the Observation object.

        :param name: The name of the observation
        :param data_set: The data_set name from which the beam data will be retrieved
        :param tx: The transmission frequency
        :return: void
        """
        self.name = name
        self.data_set = data_set
        self.data_dir = os.path.join(config.DATA_FILE_PATH, name, data_set)
        self.beam_output_data = os.path.join(config.RESULTS_FILE_PATH, name, data_set, 'beams')

        self.observation_config = self.read_xml_config()

        self.data_set_config = self._get_data_set_config(self.data_dir)

        self.tx = self.data_set_config['transmitter_frequency']

        # Read the beam data from the data_set
        beam_data = self.read_data_file(
            n_beams=self.data_set_config['nbeams'],
            n_channels=self.data_set_config['nchans'])

        # Create beams an associate them with this observation
        self.beams = self.create_beams(beam_data)

    def create_beams(self, beam_data):
        """
        Create an array of Beam objects from the beam data.

        :param beam_data: The beam_data read from the data set associated with this observation
        :return: An array of Beam objects
        """

        # Read the number of beams from the data_set config file
        # todo this must be done from the pickle file
        beams = self.observation_config.getElementsByTagName('beam')

        def attr(key):
            return beam.attributes[key].value

        beam_collection = []

        # Create beam objects for each beam in the config file
        for beam in beams:
            beam_collection.append(Beam(beam_id=attr('beamId'),
                                        dec=attr('dec'),
                                        ra=attr('ra'),
                                        ha=attr('ha'),
                                        top_frequency=attr('topFrequency'),
                                        frequency_offset=attr('frequencyOffset'),
                                        observation=self, beam_data=beam_data))

        # Return a collection of beam objects
        return beam_collection

    def read_data_file(self, n_beams, n_channels):
        """
        Read beam data from the data_set file associated with this observation
        :param n_beams: The number of beams
        :param n_channels: The number of channels
        :return: The processed beam data
        """

        file_path = self.get_data_set_file_path()

        fd = open(file_path, 'rb')
        position = 0
        no_of_doubles = 1000
        # move to position in file
        fd.seek(position, 0)

        data = np.fromfile(fd, dtype=np.dtype('f'))
        n_samples = len(data) / (n_beams * n_channels)
        data = np.reshape(data, (n_samples, n_channels, n_beams))

        # De-mirror the beam data
        # Todo - maybe this can be done at the beamforming stage
        data = self.de_mirror(data)

        return data

    def de_mirror(self, data):
        """
        De-mirror the raw beam data as read from the data set
        :param data:
        :return: The processed, de-mirrored beam data
        """

        # todo - What happens if n_sub_channels is odd? Index must be an integer
        data1 = data[:, (self.data_set_config['n_sub_channels'] * 0.5):(self.data_set_config['n_sub_channels'] - 1), :]
        data2 = data[:, (self.data_set_config['n_sub_channels'] * 1.5):, :]

        # Merge the two data set into one data set
        return np.hstack((data1, data2))

    def get_data_set_file_path(self):
        """
        Return the file path of the data set associated with this Observation
        :return: The file path of the data_set
        """

        # Retrieve the .dat file present in the data_set directory
        data_sets = [each for each in os.listdir(self.data_dir) if each.endswith('.dat')]
        return os.path.join(self.data_dir, data_sets[0])

    def _get_file_with_extension(self, base_path, extension):
        """
        Return the file with this extension in the supplied directory

        :param base_path: The base directory the file is contained in
        :param extension: The extension of the file
        :return: The full path of the file matching the extension requested
        """
        files = [each for each in os.listdir(self.data_dir) if each.endswith(extension)]
        return os.path.join(base_path, files[0])

    def _get_data_set_config(self, data_set_config_file_path):
        """
        Configure this observation with the settings of the pickle file in the data_set
        :return: void
        """
        config_file_path = self._get_file_with_extension(data_set_config_file_path, '.pkl')
        data_set_config = pickle.load(open(config_file_path, "rb"))
        # print data_set_config
        # data_set_config['nbeams'] = self.get_stop_channel()
        # data_set_config['nchans'] = data_set_config.n_channels
        # data_set_config.tx = data_set_config.transmitter_frequency

        # todo change these to the pickle file
        data_set_config['n_sub_channels'] = self.get_n_sub_channels()
        data_set_config['f_ch1'] = self.get_stop_channel()
        data_set_config['f_off'] = self.get_start_channel()
        data_set_config['sampling_rate'] = self.get_sampling_rate()

        return data_set_config

    # todo replace the below functionality with the configuration read from pickle file
    def read_xml_config(self):
        file_path = self.data_dir
        try:
            files = [each for each in os.listdir(file_path) if each.endswith('.xml')]
            xml_config = minidom.parse(os.path.join(file_path, files[0]))
            return xml_config
        except IndexError:
            log.error('XML configuration not found in' + file_path + '. Exiting.')
            exit(1)

    def get_n_beams(self):
        beams = self.observation_config.getElementsByTagName('beam')
        return len(beams)

    def get_n_channels(self):
        antenna = self.observation_config.getElementsByTagName('channels')
        return int(antenna[0].attributes['nchans'].value)

    def get_n_sub_channels(self):
        antenna = self.observation_config.getElementsByTagName('channels')
        return int(antenna[0].attributes['subchannels'].value)

    def get_sampling_rate(self):
        antenna = self.observation_config.getElementsByTagName('timing')
        return float(antenna[0].attributes['tsamp'].value)

    def get_stop_channel(self):
        antenna = self.observation_config.getElementsByTagName('channels')
        return int(antenna[0].attributes['stopChannel'].value)

    def get_start_channel(self):
        antenna = self.observation_config.getElementsByTagName('channels')
        return int(antenna[0].attributes['startChannel'].value)
