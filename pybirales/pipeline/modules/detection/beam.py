import numpy as np
from pybirales.pipeline.modules.detection.filters import RemoveBackgroundNoiseFilter, RemoveTransmitterChannelFilter, \
    PepperNoiseFilter
import warnings

from pybirales import settings

# todo - is this needed?
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
        self.tx = obs_info['transmitter_frequency']
        self.observation_id = settings.observation.id
        self.name = 'Observation ' + self.observation_name

        self._ref_time = np.datetime64(obs_info['timestamp'])
        self._time_delta = np.timedelta64(int(obs_info['sampling_time'] * 1e9), 'ns')

        self.noise = obs_info['noise']
        self.channels = channels
        self.time = time
        self.snr = beam_data[beam_id, :, :]

        self._sampling_time = obs_info['sampling_time']

    def get_config(self):
        return {
            'beam_id': self.id,
            'beam_ra': self.ra,
            'beam_dec': self.dec,
            'observation_id': self.observation_id,
            'beam_noise': self.noise,
            'tx': self.tx,
            't_0': self._ref_time,
            't_delta': self._time_delta,
            'sampling_time': self._sampling_time
        }
