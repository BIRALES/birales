import os.path
import inflection as inf
import numpy as np

from pybirales.modules.detection.filters import RemoveBackgroundNoiseFilter, RemoveTransmitterChannelFilter
from pybirales.modules.detection.repository import BeamDataRepository
from pybirales.modules.monitoring.api.common.plotters import BeamMatplotlibPlotter
from pybirales.base import settings

import warnings

warnings.filterwarnings('error')


class Beam:
    """
    The Beam class from which beam object can be created.
    """

    def __init__(self, beam_id, dec, ra, ha, top_frequency, frequency_offset, obs_info, beam_data):
        """
        Initialise the Beam class object

        :param beam_id: The beam data id
        :param dec: declination
        :param ra:
        :param ha:
        :param top_frequency:
        :param frequency_offset:
        :param obs_info:
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

        self.observation_name = settings.observation.name
        self.tx = settings.observation.transmitter_frequency
        self.n_beams = obs_info['nbeams']
        self.n_channels = obs_info['nchans']
        self.n_sub_channels = obs_info['nchans'] / 2
        self.f_ch1 = obs_info['start_center_frequency']
        self.f_off = obs_info['channel_bandwidth']
        self.sampling_rate = settings.observation.samples_per_second
        self.n_samples = obs_info['nsamp']

        self.name = self._get_human_name()
        self.time_samples = beam_data.shape[0]
        self.time = np.linspace(0, self.time_samples * self.sampling_rate, num=self.time_samples)
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
        bp.plot()

    def _get_human_name(self):
        return 'Observation ' + inf.humanize(self.observation_name)

    def _set_data(self, beam_data):
        """
        Set the beam properties (time, snr, channels) from the raw beam data (as read from data set)

        :param beam_data: The raw beam data
        :return: void
        """

        data = beam_data[0, self.id, :, :]  # beam id, channels, time samples
        # print(data)
        return self._get_snr(np.abs(data))

    def _get_snr(self, data):
        """
        Calculate the Signal to Noise Ratio from the power data

        :param data:
        :return:
        """

        # @todo - check if the mean can be used as an estimate for the noise
        mean_noise_per_channel = np.mean(data, axis=0)

        # Calculate the noise from the first 25% of the available channels
        # subset_size = int(self.n_channels * 0.25)

        # Calculate the average noise in the subset
        # noise = np.mean(data[:, range(0, subset_size)])

        indices = np.where(mean_noise_per_channel > 0)
        x = data[:, indices] / mean_noise_per_channel[indices]

        return np.log10(x)

    def _apply_filter(self, beam_filter):
        beam_filter.apply(self)

    def apply_filters(self):
        self._apply_filter(RemoveBackgroundNoiseFilter(std_threshold=2.))

        self._apply_filter(RemoveTransmitterChannelFilter())

        return self

    def save_detections(self):
        # Select points with an SNR > 0
        indices = np.where(self.snr > 0.)
        snr = self.snr[indices]
        time = self.time[indices[0]]
        channel = self.channels[indices[1]]

        detections = np.column_stack([time, channel, snr])

        repository = BeamDataRepository(self.id, self.data_set)
        repository.persist(detections)

        return indices
