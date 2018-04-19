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

        self.channel_noise = np.empty(shape=(32, 8192, settings.detection.n_noise_samples)) * np.nan

        self._observation = None

        # Flag that indicates whether the configuration was persisted
        self._config_persisted = False

        super(PreProcessor, self).__init__(config, input_blob)

        self.name = "PreProcessor"

    def _get_noise_estimation(self, power_data, iter_count):
        if self.counter < settings.detection.n_noise_samples:
            if settings.detection.noise_use_rms:
                #  use RMS
                beam_channel_noise = np.sqrt(np.mean(power_data, axis=2))
            else:
                # use mean
                beam_channel_noise = np.mean(power_data, axis=2)

            self.channel_noise[:, :, iter_count] = beam_channel_noise
        return np.nanmean(self.channel_noise, axis=2)

    def process(self, obs_info, input_data, output_data):
        """
        Filter the channelised data

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # Process only 1 polarisation
        data = input_data[0, :, :, :]

        power_data = self._power(data)

        # Estimate the noise from the data
        obs_info['channel_noise'] = self._get_noise_estimation(power_data, self.counter)
        obs_info['mean_noise'] = np.mean(obs_info['channel_noise'])

        # If the configuration was not saved AND the number of noise samples is sufficient, save the noise value.
        if not self._config_persisted and self.counter >= settings.detection.n_noise_samples:
            self._observation = Observation.objects.get(id=settings.observation.id)
            self._observation.mean_noise = obs_info['mean_noise']
            self._observation.mean_channel_noise = obs_info['channel_noise']
            self._observation.beam_noise = np.mean(obs_info['channel_noise'], axis=1)

            self._observation.save()
            self._config_persisted = True

            log.info('Mean noise {:0.3f}'.format(obs_info['mean_noise']))
        output_data[:] = power_data
        self.counter += 1

        return obs_info

    @staticmethod
    def _power(data):
        return np.power(np.abs(data), 2.0)

    @staticmethod
    def _rms(data):
        return np.sqrt(np.mean(np.power(data, 2.0)))

    def generate_output_blob(self):
        input_shape = dict(self._input.shape)

        # Generate output blob
        return ChannelisedBlob(self._config, [
            ('nbeams', input_shape['nbeams']),
            ('nchans', input_shape['nchans']),
            ('nsamp', input_shape['nsamp'])
        ], datatype=np.float)
