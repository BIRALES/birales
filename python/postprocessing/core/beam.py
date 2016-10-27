import os.path
import inflection as inf
import numpy as np

from configuration.application import config
from filters import RemoveBackgroundNoiseFilter, RemoveTransmitterChannelFilter
from helpers import LineGeneratorHelper
from visualization.api.common.plotters import BeamMatplotlibPlotter


class Beam:
    """
    The Beam class from which beam object can be created.
    """

    def __init__(self, beam_id, dec, ra, ha, top_frequency, frequency_offset, data_set, beam_data):
        """
        Initialise the Beam class object

        :param beam_id: The beam data id
        :param dec: declination
        :param ra:
        :param ha:
        :param top_frequency:
        :param frequency_offset:
        :param data_set: The DataSet object with which the beam is associated with
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

        self.observation_name = data_set.observation_name
        self.data_set = data_set
        self.tx = data_set.config['transmitter_frequency']
        self.n_beams = data_set.config['nbeams']
        self.n_channels = data_set.config['nchans']

        self.n_sub_channels = data_set.config['nchans'] / 2
        self.f_ch1 = data_set.config['f_ch1']
        self.f_off = data_set.config['f_off']
        self.sampling_rate = data_set.config['sampling_rate']
        self.n_samples = data_set.config['nsamp']
        self.name = self._get_human_name()
        self.time = np.arange(0, self.sampling_rate * self.n_samples, self.sampling_rate)
        self.channels = np.arange(self.f_ch1, self.f_ch1 + self.f_off * self.n_channels, self.f_off)
        self.snr = self._set_data(beam_data)

    def visualize(self, title):
        bp = BeamMatplotlibPlotter(fig_size=(16, 10),
                                   fig_title='Waterfall',
                                   plot_title=title,
                                   x_limits='auto',
                                   y_limits='auto',
                                   x_label='Channel',
                                   y_label='Time Sample',
                                   data=self.snr)

        file_path = os.path.join(config.get('visualization', 'FILE_PATH'), self.observation_name, self.data_set.name)
        bp.save(file_path)

    def _get_human_name(self):
        return 'Observation ' + inf.humanize(self.observation_name) + ' - ' + inf.humanize(
            os.path.basename(self.data_set.name))

    def _set_data(self, beam_data):
        """
        Set the beam properties (time, snr, channels) from the raw beam data (as read from data set)

        :param beam_data: The raw beam data
        :return: void
        """
        data = beam_data[:, :, self.id]
        return self._get_snr(data)

    @staticmethod
    def _get_snr(data):
        return np.log10(data)

    def _apply_filter(self, beam_filter):
        beam_filter.apply(self)

    def apply_filters(self):
        self._apply_filter(RemoveBackgroundNoiseFilter(std_threshold=3.))
        self._apply_filter(RemoveTransmitterChannelFilter())


class MockBeam(Beam):
    def __init__(self, beam_id, dec, ra, ha, top_frequency, frequency_offset, observation, beam_data):
        Beam.__init__(self, beam_id, dec, ra, ha, top_frequency, frequency_offset, observation, beam_data)

        self.snr = self._add_mock_tracks(self.snr)

    def _add_mock_tracks(self, snr):
        """
        Add mock detections (can be used to test pipeline)

        :param snr: The snr data (2D numpy array)
        :return: Snr data with mock tracks added
        """
        snr = self._add_mock_track(snr, (3090, 150), (3500, 200))
        snr = self._add_mock_track(snr, (2500, 90), (3000, 110))
        snr = self._add_mock_track(snr, (950, 100), (1100, 50))

        return snr

    @staticmethod
    def _add_mock_track(snr, start_coordinate=(120, 90), end_coordinate=(180, 140)):
        track_points = LineGeneratorHelper.get_line(start_coordinate, end_coordinate)  # (x0, y0), (x1, y1)
        for (channel, time) in track_points:
            snr[channel][time] += 1.0

        return snr
