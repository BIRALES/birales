import numpy as np

import struct
from matplotlib import pyplot as plt
import inflection as inf
import os.path
from app.sandbox.postprocessing.models.BeamData import BeamData
import app.sandbox.postprocessing.config.application as config


class BeamDataObservation(BeamData):
    def __init__(self, beam_id, observation):
        BeamData.__init__(self)

        self.name = 'Beam Data from Medicina'

        self.beam = beam_id
        self.observation_name = observation.name
        self.data_set = BeamDataObservation.get_data_set(config.OBSERVATION_DATA_DIR)

        self.n_beams = observation.n_beams
        self.n_channels = observation.n_channels

        self.f_ch1 = observation.f_ch1
        self.f_off = observation.f_off
        self.sampling_rate = observation.sampling_rate

        self.name = self.get_human_name()

        self.channels = None
        self.time = None
        self.snr = None

        self.set_data()

        # self.im_view()

    def get_human_name(self):
        return 'Observation ' + inf.humanize(self.observation_name) + ' - ' + inf.humanize(
                os.path.basename(self.data_set))

    def set_data(self):
        data = self.read_data_file(
                file_path = self.data_set,
                n_beams = self.n_beams,
                n_channels = self.n_channels)

        self.time = np.arange(0, self.sampling_rate * data.shape[0], self.sampling_rate)
        self.channels = (np.arange(self.f_ch1 * 1e6, self.f_ch1 * 1e6 + self.f_off * (data.shape[1]),
                                   self.f_off)) * 1e-6
        self.snr = np.log10(data[:, :, self.beam]).transpose()

    @staticmethod
    def read_data_file(file_path, n_beams, n_channels):
        # f = open(file_path, 'rb')
        #
        # data = f.read()
        # data = np.array(struct.unpack('f' * (len(data) / 4), data), dtype = float)
        # f.close()

        fd = open(file_path,'rb')
        position = 0
        no_of_doubles = 1000
        # move to position in file
        fd.seek(position, 0)

        # straight to numpy data (no buffering)
        data = np.fromfile(fd, dtype = np.dtype('f'))

        n_samples = len(data) / (n_beams * n_channels)
        data = np.reshape(data, (n_samples, n_channels, n_beams))

        return data

    def im_view(self):
        fig = plt.figure(figsize = (8, 8))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Beam %d" % self.beam)
        ax.imshow(self.snr.transpose(), aspect = 'auto',
                  origin = 'lower', extent = [self.channels[0], self.channels[-1], 0, self.time[-1]])

        ax.set_xlabel("Channel (kHz)")
        ax.set_ylabel("Time (s)")

        plt.show()

    @staticmethod
    def get_data_set(file_path):
        data_sets = [each for each in os.listdir(file_path) if each.endswith('.dat')]
        return os.path.join(file_path, data_sets[0])
