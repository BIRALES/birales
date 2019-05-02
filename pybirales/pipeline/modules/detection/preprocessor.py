import logging as log

import numpy as np

from pybirales import settings
from pybirales.pipeline.base.processing_module import ProcessingModule
from pybirales.pipeline.blobs.channelised_data import ChannelisedBlob
from pybirales.repository.models import Observation


class PreProcessor(ProcessingModule):
    _valid_input_blobs = [ChannelisedBlob]

    def __init__(self, config, input_blob=None):
        # Ensure that the input blob is of the expected format
        self._validate_data_blob(input_blob, valid_blobs=[ChannelisedBlob])

        self.counter = 0

        self._moving_avg_period = settings.detection.n_noise_samples

        self.channel_noise = np.zeros(shape=(32, 8192, self._moving_avg_period))

        self.channel_noise_std = np.zeros(shape=(32, 8192, self._moving_avg_period))

        self._observation = None

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
        self.channel_noise[:, :, iter_count % self._moving_avg_period] = np.mean(power_data, axis=2)
        self.channel_noise_std[:, :, iter_count % self._moving_avg_period] = np.std(power_data, axis=2)

        channel_noise = self.channel_noise
        channel_noise_std = self.channel_noise_std
        if self.counter < self._moving_avg_period:
            channel_noise = self.channel_noise[:,:, :self.counter+1]
            channel_noise_std = self.channel_noise_std[:,:, :self.counter+1]

        return np.mean(channel_noise, axis=2), np.mean(channel_noise_std, axis=2)

    def process(self, obs_info, input_data, output_data):
        """
        Filter the channelised data

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # Skip the first blob
        # if self._iter_count < 1:
        #     return

        # Process only 1 polarisation
        data = input_data[0, :, :, :]

        power_data = self._power(data)

        # Estimate the noise from the data (in Watts)
        # channel_noise, channel_noise_std = self._get_noise_estimation(power_data, self.counter)

        # Remove noise from power
        # power_data = power_data - channel_noise[:, :, np.newaxis]

        # Convert power data to dB
        power_data = 10*np.log10(power_data)

        # Recalculate channel noise in db
        obs_info['channel_noise'], obs_info['channel_noise_std'] = self._get_noise_estimation(power_data, self.counter)
        obs_info['mean_noise'] = np.mean(obs_info['channel_noise'])



        # print ('input', np.mean(obs_info['channel_noise']),  np.mean(obs_info['channel_noise_std']), np.mean(power_data))

        # If the configuration was not saved AND the number of noise samples is sufficient, save the noise value.
        if not self._config_persisted and self.counter >= settings.detection.n_noise_samples:
            self._observation = Observation.objects.get(id=settings.observation.id)

            self._observation.noise_mean = float(obs_info['mean_noise'])
            self._observation.noise_beams = np.mean(obs_info['channel_noise'], axis=1).tolist()

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
        # Convert to dB
        return 10 * np.log10(power)

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
        return ChannelisedBlob(self._config, [
            ('nbeams', input_shape['nbeams']),
            ('nchans', input_shape['nchans']),
            ('nsamp', input_shape['nsamp'])
        ], datatype=np.float)
