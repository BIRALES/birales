import numpy as np
import datetime
from pybirales.modules.detection.filters import RemoveBackgroundNoiseFilter, RemoveTransmitterChannelFilter, MedianFilter
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

        self.ra = settings.beamformer.pointings[self.id][0] if settings.beamformer.pointings[self.id] else 0.0
        self.dec = settings.beamformer.pointings[self.id][1] if settings.beamformer.pointings[self.id] else 0.0

        self.ha = 0.0
        self.top_frequency = 0.0
        self.frequency_offset = 0.0

        self.observation_name = settings.observation.name
        self.tx = settings.observation.transmitter_frequency
        self.configuration_id = obs_info['configuration_id']
        self.n_beams = obs_info['nbeams']
        self.n_channels = obs_info['nchans']
        self.n_sub_channels = obs_info['nchans'] / 2
        self.f_ch1 = obs_info['start_center_frequency']
        self.f_off = obs_info['channel_bandwidth']
        self.sampling_rate = settings.observation.samples_per_second
        self.n_samples = obs_info['nsamp']

        self.name = 'Observation ' + self.observation_name

        self.t_0 = obs_info['timestamp']
        self.dt = obs_info['sampling_time']
        self.time = np.arange(0, beam_data.shape[3])

        self.channels = np.arange(self.f_ch1, self.f_ch1 + self.f_off * self.n_channels, self.f_off)
        self.snr = self._set_snr(beam_data)

    def _rms(self, data):
        return np.sqrt(np.mean(data**2.0))

    def _power(self, data):
        return np.abs(data) ** 2

    def _set_snr(self, data):
        """
        Calculate the Signal to Noise Ratio from the power data

        :param data:
        :return:
        """

        # return np.abs(data[0, self.id, int(self.n_channels / 2):, :]).T
        # data = np.abs(data[0, self.id, :, :]).T
        #
        # @todo - check if the mean can be used as an estimate for the noise
        # mean_noise_per_channel = np.mean(data, axis=0)
        #
        # # Normalised the data by the mean noise at each channel
        # normalised_data = np.where(data > 0., data, np.nan) / mean_noise_per_channel

        data = data[0, self.id, :, :].T

        # version 2 - start
        p_v = self._power(data)
        # p_n = self._power(np.mean(data, axis=0))
        p_n = self._power(self._rms(data))
        snr = (p_v - p_n) / p_n
        snr[snr <= 0] = np.nan
        log_data = 10 * np.log10(snr)
        log_data[np.isnan(log_data)] = 0.
        # version 2 - end

        return log_data

    def _apply_filter(self, beam_filter):
        beam_filter.apply(self)

    def apply_filters(self):
        # Remove background noise
        self._apply_filter(RemoveBackgroundNoiseFilter(std_threshold=2.))

        # Remove transmitter frequency
        self._apply_filter(RemoveTransmitterChannelFilter())

        # Remove transmitter frequency
        self._apply_filter(MedianFilter())

        return self
