import numpy as np

from pybirales.modules.detection.filters import RemoveBackgroundNoiseFilter, RemoveTransmitterChannelFilter
from pybirales.modules.detection.repository import BeamDataRepository
from pybirales.base import settings
import warnings

warnings.filterwarnings('error')


class Beam:
    """
    The Beam class from which beam object can be created.
    """

    def __init__(self, beam_id, obs_info, beam_data):
        """
        Initialise the Beam class object

        :param beam_id: The beam data id
        :param obs_info:
        :param beam_data:
        :return: void
        """
        self.id = beam_id
        self.name = 'Beam ' + str(beam_id)

        self.dec = 0.0
        self.ra = 0.0
        self.ha = 0.0
        self.top_frequency = 0.0
        self.frequency_offset = 0.0

        self.observation_name = settings.observation.name
        self.tx = settings.observation.transmitter_frequency
        self.n_beams = obs_info['nbeams']
        self.n_channels = obs_info['nchans']
        self.n_sub_channels = obs_info['nchans'] / 2
        self.f_ch1 = obs_info['start_center_frequency']
        self.f_off = obs_info['channel_bandwidth']
        self.sampling_rate = settings.observation.samples_per_second
        self.n_samples = obs_info['nsamp']

        self.name = 'Observation ' + self.observation_name
        self.time_samples = beam_data.shape[3]
        self.time = np.linspace(0, self.time_samples * self.sampling_rate, num=self.time_samples)
        self.channels = np.arange(self.f_ch1, self.f_ch1 + self.f_off * self.n_channels, self.f_off)
        self.snr = self._set_snr(beam_data)

    def _set_snr(self, data):
        """
        Calculate the Signal to Noise Ratio from the power data

        :param data:
        :return:
        """

        return np.abs(data[0, self.id, 0:int(self.n_channels / 2), :]).T

        #
        # # @todo - check if the mean can be used as an estimate for the noise
        # mean_noise_per_channel = np.mean(data, axis=0)
        #
        # # Normalised the data by the mean noise at each channel
        # normalised_data = np.where(data > 0., data, np.nan) / mean_noise_per_channel
        #
        # # Take the log value of the power
        # log_data = np.log10(normalised_data)
        #
        # # Replace nan values with 0.
        # log_data[np.isnan(log_data)] = 0.
        #
        # return log_data

    def _apply_filter(self, beam_filter):
        beam_filter.apply(self)

    def apply_filters(self):
        # Remove background noise
        self._apply_filter(RemoveBackgroundNoiseFilter(std_threshold=2.))

        # Remove transmitter frequency
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
