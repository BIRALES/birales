import logging as log

import numpy as np

from pybirales import settings
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob, GPUChannelisedBlob
from pybirales.pipeline.modules.detection.util import apply_doppler_mask
from pybirales.repository.models import Observation

cu = None
try:
    import cupy as cu
except ImportError:
    pass


class PreProcessor(ProcessingModule):

    def __init__(self, config, input_blob=None):
        # Ensure that the input blob is of the expected format
        self._validate_data_blob(input_blob, valid_blobs=[ChannelisedBlob, GPUChannelisedBlob])

        self._copy_from_gpu = False
        if isinstance(input_blob, GPUChannelisedBlob):
            self._copy_from_gpu = True

        self.counter = 0

        self._moving_avg_period = settings.detection.n_noise_samples

        self._n_channels = 3758  # new tpm

        self.channel_noise = None

        self.channel_noise_std = None

        self._observation = None

        self.channels = None
        self._doppler_mask = None

        # Flag that indicates whether the configuration was persisted
        self._config_persisted = False

        super(PreProcessor, self).__init__(config, input_blob)

        self.name = "PreProcessor"

    def _get_noise_estimation(self, power_data, iter_count):
        """

        :param power_data:
        :param iter_count:
        :return:
        """
        # print power_data.shape
        self.channel_noise[:, :, iter_count % self._moving_avg_period] = np.median(power_data, axis=2)
        self.channel_noise_std[:, :, iter_count % self._moving_avg_period] = np.std(power_data, axis=2)

        channel_noise = self.channel_noise
        channel_noise_std = self.channel_noise_std
        if self.counter < self._moving_avg_period:
            channel_noise = self.channel_noise[:, :, :self.counter + 1]
            channel_noise_std = self.channel_noise_std[:, :, :self.counter + 1]

        return np.median(channel_noise, axis=2), np.median(channel_noise_std, axis=2)

    def process(self, obs_info, input_data, output_data):
        """
        Filter the channelised data

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # Skip the first blob
        if self._iteration_counter < 0:
            return

        self.channels, self._doppler_mask = apply_doppler_mask(self._doppler_mask, self.channels,
                                                               settings.detection.doppler_range,
                                                               obs_info)

        if self.channel_noise is None:
            self.channel_noise = np.zeros(shape=(obs_info['nof_beams'], len(self.channels),
                                                 self._moving_avg_period))
            self.channel_noise_std = np.zeros(shape=(obs_info['nof_beams'], len(self.channels),
                                                     self._moving_avg_period))

        if not 'doppler_mask' in obs_info:
            obs_info['doppler_mask'] = self._doppler_mask
            obs_info['channels'] = self.channels

        # If the data is coming from a GPU blob, then copy to host first
        if self._copy_from_gpu:
            data = cu.asnumpy(input_data[0][:, self._doppler_mask, :])
        else:
            data = input_data[0][:, self._doppler_mask, :]

        power_data = self._power(data)

        # Recalculate channel noise in db
        obs_info['channel_noise'], obs_info['channel_noise_std'] = self._get_noise_estimation(power_data, self.counter)
        obs_info['mean_noise'] = np.median(obs_info['channel_noise'])

        print('>', self.counter, np.max(power_data), np.argmax(power_data), np.max(obs_info['channel_noise'][15]),
              np.argmax(obs_info['channel_noise'][15]))

        power_data = power_data - obs_info['channel_noise'][..., np.newaxis]

        # If the configuration was not saved AND the number of noise samples is sufficient, save the noise value.
        if not self._config_persisted and self.counter >= settings.detection.n_noise_samples and settings.database.load_database:
            self._observation = Observation.objects.get(id=settings.observation.id)

            self._observation.noise_mean = float(obs_info['mean_noise'])
            self._observation.noise_beams = np.median(obs_info['channel_noise'], axis=1).tolist()

            self._observation.tx = obs_info['transmitter_frequency']
            self._observation.sampling_time = obs_info['sampling_time']

            for h in log.getLoggerClass().root.handlers:
                try:
                    self._observation.log_filepath = h.baseFilename
                except AttributeError:
                    continue

            self._observation.save()
            self._config_persisted = True

            log.info('Mean noise {:0.3f}'.format(obs_info['mean_noise']))

        output_data[:] = power_data

        self.counter += 1

        return obs_info

    # @staticmethod
    def _power(self, data):
        """
        Calculate the power from the input data
        :param data:
        :return:
        """
        power = np.power(np.abs(data), 2.0) + 0.00000000000001

        return power

    @staticmethod
    def _rms(data):
        """
        Calculate the rms from the input data

        :param data:
        :return:
        """
        return np.sqrt(np.mean(np.power(data, 2.0)))

    def generate_output_blob(self):
        input_shape = dict(self._input.shape)

        # Generate output blob
        return ChannelisedBlob([
            ('nof_beams', input_shape['nof_beams']),
            ('nof_channels', self._n_channels),
            ('nof_samples', input_shape['nof_samples'])
        ], datatype=float)
