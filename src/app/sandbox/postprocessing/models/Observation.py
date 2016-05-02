import os
from xml.dom import minidom
import app.sandbox.postprocessing.config.application as config
from app.sandbox.postprocessing.models.Beam import Beam
import numpy as np


class Observation:
    def __init__(self, name, data_set, tx):
        self.name = name
        self.data_set = data_set
        self.data_dir = os.path.join(config.DATA_FILE_PATH, name, data_set)
        self.beam_output_data = os.path.join(config.RESULTS_FILE_PATH, name, data_set, 'beams')

        self.observation_config = self.read_xml_config()

        # Read from data_set's xml configuration file
        self.n_beams = self.get_n_beams()
        self.n_channels = self.get_n_channels()
        self.n_sub_channels = self.get_n_sub_channels()
        self.tx = tx

        self.f_ch1 = self.get_stop_channel()
        self.f_off = -19531.25  # 20Mhz / 1024
        self.f_off = self.get_start_channel()
        self.sampling_rate = self.get_sampling_rate()

        beam_data = self.read_data_file(

                n_beams = self.n_beams,
                n_channels = self.n_sub_channels * 2)

        # init beams in observation
        self.beams = self.create_beams(beam_data)

    def create_beams(self, beam_data):
        beams = self.observation_config.getElementsByTagName('beam')

        def attr(key):
            return beam.attributes[key].value

        beam_collection = []
        for beam in beams:
            beam_collection.append(Beam(beam_id = attr('beamId'),
                                        dec = attr('dec'),
                                        ra = attr('ra'),
                                        ha = attr('ha'),
                                        top_frequency = attr('topFrequency'),
                                        frequency_offset = attr('frequencyOffset'),
                                        observation = self, beam_data = beam_data))

        return beam_collection

    def read_data_file(self, n_beams, n_channels):
        file_path = self.get_data_set_file_path()

        fd = open(file_path, 'rb')
        position = 0
        no_of_doubles = 1000
        # move to position in file
        fd.seek(position, 0)

        data = np.fromfile(fd, dtype = np.dtype('f'))
        n_samples = len(data) / (n_beams * n_channels)
        data = np.reshape(data, (n_samples, n_channels, n_beams))

        data = self.de_mirror(data)

        return data

    def de_mirror(self, data):
        data1 = data[:, (self.n_sub_channels * 0.5):(self.n_sub_channels - 1), :]
        data2 = data[:, (self.n_sub_channels * 1.5):, :]

        return np.hstack((data1, data2))

    def get_data_set_file_path(self):
        data_sets = [each for each in os.listdir(self.data_dir) if each.endswith('.dat')]
        return os.path.join(self.data_dir, data_sets[0])

    def read_xml_config(self):
        file_path = self.data_dir
        files = [each for each in os.listdir(file_path) if each.endswith('.xml')]
        xml_config = minidom.parse(os.path.join(file_path, files[0]))

        return xml_config

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
