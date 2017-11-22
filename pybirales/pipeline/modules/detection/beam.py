import numpy as np
import datetime
from pybirales.pipeline.modules.detection.filters import RemoveBackgroundNoiseFilter, RemoveTransmitterChannelFilter, \
    PepperNoiseFilter
from pybirales import settings
import warnings

warnings.filterwarnings('error')


class Beam:
    """
    The Beam class from which beam object can be created.
    """

    def __init__(self, beam_id, obs_info, channels, time, beam_data):
        """
        Initialise the Beam class object

        :param beam_id: The beam data id
        :param obs_info:
        :param beam_data:
        :return: void
        """
        self.id = beam_id
        self.name = 'Beam ' + str(beam_id)

        self.ra = obs_info['pointings'][self.id][0] if obs_info['pointings'][self.id] else 0.0
        self.dec = obs_info['pointings'][self.id][1] if obs_info['pointings'][self.id] else 0.0

        self.observation_name = obs_info['settings']['observation']['name']
        self.tx = settings.observation.transmitter_frequency
        self.configuration_id = obs_info['configuration_id']
        self.name = 'Observation ' + self.observation_name

        self.noise = obs_info['noise']
        self.channels = channels
        self.time = time
        self.snr = self._set_snr(beam_data)

    @staticmethod
    def _rms(data):
        return np.sqrt(np.mean(np.power(data, 2.0)))

    @staticmethod
    def _power(data):
        return np.power(np.abs(data), 2.0)

    def _set_snr(self, data):
        """
        Calculate the Signal to Noise Ratio from the power data

        :param data:
        :return:
        """

        data = data[0, self.id, :, :].T
        # version 3 - start
        p_v = self._power(data)
        p_n = self.noise
        snr = p_v / p_n
        snr[snr <= 0] = np.nan
        log_data = 10 * np.log10(snr)
        log_data[np.isnan(log_data)] = 0.
        # version 3 - end

        return log_data

    def _apply_filter(self, beam_filter):
        beam_filter.apply(self)

    def apply_filters(self):
        # Remove background noise
        self._apply_filter(RemoveBackgroundNoiseFilter(std_threshold=2.))

        # Remove transmitter frequency
        self._apply_filter(RemoveTransmitterChannelFilter())

        # Remove pepper noise from the data
        self._apply_filter(PepperNoiseFilter())

        return self

    def get_config(self):
        return {
            'beam_id': self.id,
            'beam_ra': self.ra,
            'beam_dec': self.dec,
            'configuration_id': self.configuration_id,
            'beam_noise': self.noise,
            'tx': self.tx,
        }
