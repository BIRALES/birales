import numpy as np

import struct
from matplotlib import pyplot as plt
import inflection as inf
import os.path
from app.sandbox.postprocessing.models.BeamData import BeamData


class BeamDataObservation(BeamData):
    def __init__(self, n_beams = 32, n_channels = 8192, beam = 15, f_ch1 = 418, f_off = -19531.25,
                 sampling_rate = 0.0000512):
        BeamData.__init__(self)

        self.name = 'Beam Data from Medicina'
        self.repository = '/home/denis/dev/birales/src/app/sandbox/postprocessing/data/'
        self.observation = 'medicina_07_03_2016/'
        self.data_file = '40058/40058_2016-03-08_15:09:40.dat'

        self.n_beams = n_beams
        self.n_channels = n_channels
        self.beam = beam
        self.f_ch1 = f_ch1
        self.f_off = f_off
        self.sampling_rate = sampling_rate

        self.name = 'Observation ' + inf.humanize(self.observation) + ' - ' + inf.humanize(
                os.path.basename(self.data_file))

        self.channels = None
        self.time = None
        self.snr = None

        self.set_data()

        self.im_view()

    def set_data(self):
        data = self.read_data_file(
                file_path = os.path.join(self.repository, self.observation, self.data_file),
                n_beams = self.n_beams,
                n_channels = self.n_channels)

        self.time = np.arange(0, self.sampling_rate * data.shape[0], self.sampling_rate)
        self.channels = (
                        np.arange(self.f_ch1 * 1e6, self.f_ch1 * 1e6 + self.f_off * (data.shape[1]), self.f_off)) * 1e-6
        self.snr = np.log10(data[:, :, self.beam]).transpose()

    @staticmethod
    def read_data_file(file_path, n_beams, n_channels):
        f = open(file_path, 'rb')
        data = f.read()
        f.close()

        data = np.array(struct.unpack('f' * (len(data) / 4), data), dtype = float)
        n_samples = len(data) / (n_beams * n_channels)
        data = np.reshape(data, (n_samples, n_channels, n_beams))

        return data

    def im_view(self):
        fig = plt.figure(figsize = (8, 8))

        ax = fig.add_subplot(1, 2, 1)
        ax.set_title("Beam %d" % self.beam)
        ax.imshow(self.snr.transpose(), aspect = 'auto',
                  origin = 'lower', extent = [self.channels[0], self.channels[-1], 0, self.time[-1]])

        ax.set_xlabel("Channel (kHz)")
        ax.set_ylabel("Time (s)")

        plt.show()
