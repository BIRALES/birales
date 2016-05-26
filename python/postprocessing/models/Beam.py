import os.path
import numpy as np
import inflection as inf

from helpers.LineGeneratorHelper import LineGeneratorHelper
from matplotlib import pyplot as plt
from views.BeamDataView import BeamDataView


class Beam:
    """
    The Beam class from which beam object can be created.
    """

    def __init__(self, beam_id, dec, ra, ha, top_frequency, frequency_offset, observation, beam_data):
        """

        :param beam_id: The beam data id
        :param dec: declination
        :param ra:
        :param ha:
        :param top_frequency:
        :param frequency_offset:
        :param observation: The observation object with which the beam is associated with
        :param beam_data:
        :return: void
        """
        self.id = beam_id
        self.name = 'Beam ' + str(beam_id)

        self.dec = dec
        self.ra = ra
        self.ha = ha
        self.top_frequency = top_frequency
        self.frequency_offset = frequency_offset
        self._view = BeamDataView()

        self.observation = observation  # todo - check if the dependency on observation can be removed.

        self.observation_name = observation.name
        self.data_set = Beam.get_data_set(observation.data_dir)
        self.tx = observation.data_set_config['transmitter_frequency']
        self.n_beams = observation.data_set_config['nbeams']
        self.n_channels = observation.data_set_config['nchans']

        self.n_sub_channels = observation.data_set_config['nchans'] / 2
        self.f_ch1 = observation.data_set_config['f_ch1']
        self.f_off = observation.data_set_config['f_off']
        self.sampling_rate = observation.data_set_config['sampling_rate']

        self.name = self.get_human_name()

        self.channels = None
        self.time = None
        self.snr = None

        self.set_data(beam_data, self.id)
        # if self.id == 15:
        #     self.im_view()

    def save(self, file_name):
        beam_id = 'beam_' + str(self.id)

        file_path = os.path.join(self.observation.beam_output_data, beam_id, file_name)
        self.view(file_path)

    def get_human_name(self):
        return 'Observation ' + inf.humanize(self.observation_name) + ' - ' + inf.humanize(
            os.path.basename(self.data_set))

    # todo remove beam id from this use self.id
    def set_data(self, beam_data, beam_id):
        """
        Set the beam properties (time, snr, channels) from the raw beam data (as read from data set)

        :param beam_data: The raw beam data
        :param beam_id: The id of this beam
        :return: void
        """
        data = beam_data[:, :, beam_id]
        self.time = np.arange(0, self.sampling_rate * data.shape[0], self.sampling_rate)

        start = self.f_ch1 - ((20. / self.n_channels) * self.f_off)
        rate = ((20. / self.n_channels) / self.n_sub_channels) * 2
        # rate = 20. / self.n_channels
        # self.channels = np.arange(start, start + rate * self.n_channels, rate)
        self.channels = np.arange(start, start + rate * self.n_sub_channels * 2, rate)

        self.time = np.arange(0, data.shape[0], 1)
        self.channels = np.arange(0, data.shape[1], 1)

        self.time = np.arange(0, self.sampling_rate * data.shape[0], self.sampling_rate)
        self.channels = np.arange(self.f_ch1, self.f_ch1 + self.f_off * self.n_channels, self.f_off)

        self.snr = np.log10(data).transpose()

        # self.im_view()
        # exit(0)
        # self.snr = self.add_mock_tracks(self.snr)

    @staticmethod
    def add_mock_tracks(snr):
        """
        Add mock detections (can be used to test pipeline)

        :param snr: The snr data (2D numpy array)
        :return: Snr data with mock tracks added
        """
        snr = Beam.add_mock_track(snr, (3090, 150), (3500, 200))
        snr = Beam.add_mock_track(snr, (2500, 90), (3000, 110))
        snr = Beam.add_mock_track(snr, (950, 100), (1100, 50))

        return snr

    def im_view(self):
        """
        Naive plotting of the raw data. This is usually used to visualise quickly the beam data read
        :return: void
        """

        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Beam %s" % self.id)
        ax.imshow(self.snr.transpose(), aspect='auto',
                  origin='lower', extent=[self.channels[0], self.channels[-1], 0, self.time[-1]])

        ax.set_xlabel("Channel (kHz)")
        ax.set_ylabel("Time (s)")

        plt.show()

    @staticmethod
    def get_data_set(file_path):
        """
        Return the file path of the data set associated with this Observation
        :param: The file path of the folder which co
        :return: The file path of the data_set
        """
        data_sets = [each for each in os.listdir(file_path) if each.endswith('.dat')]
        return os.path.join(file_path, data_sets[0])

    @staticmethod
    def add_mock_track(snr, start_coordinate=(120, 90), end_coordinate=(180, 140)):
        track_points = LineGeneratorHelper.get_line(start_coordinate, end_coordinate)  # (x0, y0), (x1, y1)
        mean = np.mean(snr)
        for (channel, time) in track_points:
            snr[channel][time] += 1.0

        return snr

    def view(self, file_path, name='Beam Data'):
        if name:
            self._view.name = name

        self._view.set_layout(figure_title=name,
                              x_axis_title='Frequency (Hz)',
                              y_axis_title='Time (s)')

        self._view.set_data(data=self.snr.transpose(),
                            x_axis=self.channels,
                            y_axis=self.time)

        self._view.save(file_path)

    def get_view(self):
        return self._view
