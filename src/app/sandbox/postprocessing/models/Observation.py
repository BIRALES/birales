import os
from xml.dom import minidom
import app.sandbox.postprocessing.config.application as config


class Observation:

    def __init__(self, name, data_set):
        self.name = name
        self.data_set = data_set
        self.data_dir = os.path.join(config.DATA_FILE_PATH, name, data_set)
        self.beam_output_data = os.path.join(config.RESULTS_FILE_PATH, name, data_set, 'beams')

        self.observation_config = self.read_xml_config()

        # Read from data_set's xml configuration file
        self.n_beams = self.get_n_beams()
        self.n_channels = self.get_n_channels()
        self.n_sub_channels = self.get_n_sub_channels()
        self.tx = 150

        self.f_ch1 = self.get_stop_channel()
        self.f_off = -19531.25  # 20Mhz / 1024
        self.f_off = self.get_start_channel()
        self.sampling_rate = self.get_sampling_rate()

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