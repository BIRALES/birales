import numpy as np
import struct
from matplotlib import pyplot as plt
import inflection as inf
import os.path

from app.sandbox.postprocessing.models.BeamData import BeamData
from app.sandbox.postprocessing.helpers.LineGeneratorHelper import LineGeneratorHelper
import logging as log


class BeamDataObservation(BeamData):
    def __init__(self, beam_id, observation):
        BeamData.__init__(self)

        self.name = 'Beam Data from Medicina'

        self.beam = beam_id
        self.observation_name = observation.name
        self.data_set = BeamDataObservation.get_data_set(observation.data_dir)

        self.n_beams = observation.n_beams
        self.n_channels = observation.n_channels
        self.n_sub_channels = observation.n_sub_channels

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
        log.info('Reading data file %s', self.data_set)
        data = self.read_data_file(
                file_path = self.data_set,
                n_beams = self.n_beams,
                n_channels = self.n_sub_channels * 2)

        data = self.de_mirror(data)


        self.time = np.arange(0, self.sampling_rate * data.shape[0], self.sampling_rate)
        # self.channels = (np.arange(self.f_ch1 * 1e6, self.f_ch1 * 1e6 + self.f_off * (data.shape[1]),
        #                            self.f_off)) * 1e-6

        start = self.f_ch1 - ((20. / self.n_channels) * self.f_off)
        rate = ((20. / self.n_channels) / self.n_sub_channels) * 2

        self.channels = np.arange(start, start + rate * self.n_sub_channels * 2, rate)
        # self.time = np.arange(0, data.shape[0], 1)
        # self.channels = np.arange(0, data.shape[1], 1)
        self.snr = np.log10(data).transpose()

        # self.snr = self.add_mock_tracks(self.snr)

    @staticmethod
    def add_mock_tracks(snr):
        snr = BeamDataObservation.add_mock_track(snr, (3090, 150), (3500, 200))
        snr = BeamDataObservation.add_mock_track(snr, (2500, 90), (3000, 110))
        snr = BeamDataObservation.add_mock_track(snr, (950, 100), (1100, 50))

        return snr

    def de_mirror(self, data):
        data1 = data[:, (self.n_sub_channels * 0.5):(self.n_sub_channels - 1), self.beam]
        data2 = data[:, (self.n_sub_channels * 1.5):, self.beam]

        data = np.hstack((data1, data2))
        print data.shape
        exit(0)
        return data

    @staticmethod
    def read_data_file(file_path, n_beams, n_channels):
        # f = open(file_path, 'rb')
        #
        # data = f.read()
        # data = np.array(struct.unpack('f' * (len(data) / 4), data), dtype = float)
        # f.close()

        fd = open(file_path, 'rb')
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

    @staticmethod
    def add_mock_track(snr, start_coordinate = (120, 90), end_coordinate = (180, 140)):
        track_points = LineGeneratorHelper.get_line(start_coordinate, end_coordinate)  # (x0, y0), (x1, y1)
        mean = np.mean(snr)
        for (channel, time) in track_points:
            snr[channel][time] += 1.0

        return snr
