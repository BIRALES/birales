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

        self.noise = []

        self._observation = None

        # Flag that indicates whether the configuration was persisted
        self._config_persisted = False

        super(PreProcessor, self).__init__(config, input_blob)

        self.name = "PreProcessor"

    def _get_noise_estimation(self, data):
        if self.counter < settings.detection.n_noise_samples:
            power = np.power(np.abs(data[:, settings.detection.noise_channels, :]), 2)

            if settings.detection.noise_use_rms:
                #  use RMS
                noise = np.sqrt(np.mean(np.power(power, 2)))
            else:
                # use mean
                noise = np.mean(power)

            self.noise.append(noise)
            self.mean_noise = np.mean(self.noise)
        return float(self.mean_noise)

    def process(self, obs_info, input_data, output_data):
        """
        Filter the channelised data

        :param obs_info:
        :param input_data:
        :param output_data:
        :return:
        """

        # Reduce the blob to a single polarization
        data = input_data[0, :, :, :]

        # Estimate the noise from the data
        obs_info['noise'] = self._get_noise_estimation(data)

        # If the configuration was not saved AND the number of noise samples is sufficient, save the noise value.
        if not self._config_persisted and self.counter >= settings.detection.n_noise_samples:
            self._observation = Observation.objects.get(id=settings.observation.id)
            self._observation.noise_estimate = obs_info['noise']
            self._observation.save()
            self._config_persisted = True

        # version 3 - start
        p_v = self._power(data)
        p_n = obs_info['noise']
        snr = p_v / p_n
        snr[snr <= 0] = np.nan
        log_data = 10 * np.log10(snr)
        log_data[np.isnan(log_data)] = 0.

        output_data[:] = log_data
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
