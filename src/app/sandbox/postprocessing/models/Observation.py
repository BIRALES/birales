import os
from xml.dom import minidom
import app.sandbox.postprocessing.config.application as config


class Observation:

    def __init__(self, name, data_set):
        self.name = name
        self.data_set = data_set
        # self.beam = 15  # todo make configurable

        self.observation_config = Observation.read_xml_config()

        # Read from data_set's xml configuration file
        self.n_beams = self.get_n_beams()
        self.n_channels = self.get_n_channels()
        self.tx = 150

        self.f_ch1 = self.get_stop_channel()
        self.f_off = -19531.25  # todo check from where to get this
        self.sampling_rate = self.get_sampling_rate()

    @staticmethod
    def read_xml_config():
        file_path = config.OBSERVATION_DATA_DIR
        files = [each for each in os.listdir(file_path) if each.endswith('.xml')]
        xml_config = minidom.parse(os.path.join(file_path, files[0]))

        return xml_config

    def get_n_beams(self):
        beams = self.observation_config.getElementsByTagName('beam')
        return len(beams)

    def get_n_channels(self):
        antenna = self.observation_config.getElementsByTagName('samples')
        return int(antenna[0].attributes['nsamp'].value)

    def get_sampling_rate(self):
        antenna = self.observation_config.getElementsByTagName('timing')
        return float(antenna[0].attributes['tsamp'].value)

    def get_stop_channel(self):
        antenna = self.observation_config.getElementsByTagName('channels')
        return int(antenna[0].attributes['stopChannel'].value)